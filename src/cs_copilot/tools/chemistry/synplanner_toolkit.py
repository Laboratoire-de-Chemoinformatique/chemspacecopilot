#!/usr/bin/env python
# coding: utf-8
"""Integration with the official SynPlanner retrosynthesis package.

This module exposes a toolkit that mirrors the workflow demonstrated in the
public SynPlanner Colab notebook.  Rather than providing a heuristic
approximation, the toolkit wraps the real ``SynPlanner`` Python package and
executes the same high-level steps that the notebook follows:

1. Load SynPlanner components (reaction rules, building blocks, policy network)
   from the data folder (downloading if necessary).
2. Normalise the user input, accepting either SMILES strings or trivial
   molecule names and resolving them to canonical SMILES.
3. Create a Tree with TreeConfig and run the search using PolicyNetworkFunction.
4. Extract routes from the Tree's winning_nodes and post-process them into
   structured summaries that agents can consume.

When the ``SynPlanner`` dependency is missing, the toolkit raises a helpful
exception explaining how to install it.  This mirrors the behaviour one would
see when running the notebook without first installing the package.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agno.agent import Agent

from cs_copilot.storage import S3, OutputOperation, operation_rel_path
from cs_copilot.tools.io.formatting import smiles_to_png_bytes
from cs_copilot.tools.io.session_memory import (
    list_session_objects,
    register_session_object,
    update_session_object,
    update_state_targets,
)

from .base_chemistry import BaseChemistryToolkit, InvalidSMILESError
from .standardize import standardize_smiles

logger = logging.getLogger(__name__)


def _session_state_for_outputs(
    agent: Optional[Agent],
    session_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if isinstance(session_state, dict):
        return session_state
    agent_state = getattr(agent, "session_state", None)
    return agent_state if isinstance(agent_state, dict) else None


def _target_slug(query: Optional[str], smiles: Optional[str]) -> str:
    seed = str(smiles or query or "target")
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(query or "target")).strip("._-")
    label = label[:48].strip("._-") or "target"
    return f"{label}_{digest}"


class SynPlannerError(Exception):
    """Raised when the SynPlanner backend cannot be used."""


class UserConfirmationRequiredError(SynPlannerError):
    """Raised when user confirmation is needed for a SMILES string.

    This exception contains the SMILES string and image data that should be
    displayed to the user for confirmation.
    """

    def __init__(self, message: str, smiles: str, image_data: bytes, molecule_name: str):
        super().__init__(message)
        self.smiles = smiles
        self.image_data = image_data
        self.molecule_name = molecule_name


@dataclass
class _NormalisedStep:
    """Internal representation of a retrosynthetic step."""

    index: int
    description: str
    reactants: Sequence[str]
    products: Sequence[str]
    reagents: Sequence[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "reactants": list(self.reactants),
            "products": list(self.products),
            "reagents": list(self.reagents),
        }


@dataclass(frozen=True)
class _SearchProfile:
    """SynPlanner search configuration for one planning attempt."""

    name: str
    tree_config: Dict[str, Any]
    policy_config: Dict[str, Any]


class SynPlannerToolkit(BaseChemistryToolkit):
    """Expose SynPlanner retrosynthesis routines as a toolkit."""

    #: Minimal dictionary for offline name-to-SMILES resolution.  The official
    #: package performs the resolution via PubChem in the notebook; the
    #: dictionary keeps the toolkit usable in offline CI environments.
    _FALLBACK_NAMES: Dict[str, str] = {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "paracetamol": "CC(=O)NC1=CC=C(O)C=C1O",
        "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    }

    def __init__(
        self,
        *,
        prefer_gpu: bool = False,
        default_top_k: int = 3,
        data_folder: Optional[str] = None,
        max_iterations: int = 100,
        max_tree_size: int = 10000,
        max_time: int = 120,
        max_depth: int = 9,
        min_mol_size: int = 6,
        top_rules: int = 50,
        rule_prob_threshold: float = 0.0,
        enable_retry_profiles: bool = True,
    ) -> None:
        """Initialise the toolkit and register exposed tools.

        Args:
            prefer_gpu: Whether to prefer GPU execution (not currently used by SynPlanner Tree)
            default_top_k: Default number of routes to return
            data_folder: Path to SynPlanner data folder (if None, will try to use default locations)
            max_iterations: Maximum number of search iterations for Tree
            max_tree_size: Maximum number of nodes in the search tree
            max_time: Maximum search time in seconds for Tree
            max_depth: Maximum depth of the search tree
            min_mol_size: Molecules at or below this size are treated as building blocks
            top_rules: Number of top policy-network reaction rules to consider
            rule_prob_threshold: Minimum policy probability for selected reaction rules
            enable_retry_profiles: Whether to retry with broader profiles when no routes are found
        """

        super().__init__(name="synplanner")
        self.prefer_gpu = prefer_gpu
        self.default_top_k = default_top_k
        self.data_folder = data_folder
        self.max_iterations = max_iterations
        self.max_tree_size = max_tree_size
        self.max_time = max_time
        self.max_depth = max_depth
        self.min_mol_size = min_mol_size
        self.top_rules = top_rules
        self.rule_prob_threshold = rule_prob_threshold
        self.enable_retry_profiles = enable_retry_profiles
        self._synplanner_module: Optional[Any] = None
        self._reaction_rules: Optional[Any] = None
        self._building_blocks: Optional[Any] = None
        self._policy_network: Optional[Any] = None
        self._last_plan: Optional[Dict[str, Any]] = None

        # Register public tools for the agent framework.
        self.register(self.identify_input)
        self.register(self.convert_name_to_smiles)
        self.register(self.plan_synthesis)
        self.register(self.describe_plan)
        self.register(self.get_route_visualizations)

    # ------------------------------------------------------------------
    # SynPlanner backend loading
    # ------------------------------------------------------------------
    def _import_synplanner(self) -> Any:
        if self._synplanner_module is not None:
            return self._synplanner_module

        try:
            module = importlib.import_module("synplan")
        except ImportError as exc:  # pragma: no cover - defensive branch
            raise SynPlannerError(
                "The 'synplanner' package is required. Install it with 'pip install SynPlanner'."
            ) from exc

        self._synplanner_module = module
        return module

    def _load_synplanner_components(self) -> None:
        """Load SynPlanner components (reaction rules, building blocks, policy network)."""
        if (
            self._reaction_rules is not None
            and self._building_blocks is not None
            and self._policy_network is not None
        ):
            return

        self._import_synplanner()

        # Import required modules
        try:
            from pathlib import Path

            from synplan.mcts.expansion import PolicyNetworkFunction
            from synplan.utils.config import PolicyNetworkConfig
            from synplan.utils.loading import (
                download_all_data,
                load_building_blocks,
                load_reaction_rules,
            )
        except ImportError as exc:
            raise SynPlannerError(
                f"Failed to import SynPlanner components: {exc}. "
                "Make sure SynPlanner is properly installed."
            ) from exc

        # Determine data folder
        data_folder = None
        if self.data_folder:
            data_folder = Path(self.data_folder)
        else:
            # Find project root by looking for pyproject.toml or synplan_data
            project_root = None
            current = Path(__file__).parent
            # Walk up from the current file location to find project root
            for parent in [current] + list(current.parents):
                if (parent / "pyproject.toml").exists() or (parent / "synplan_data").exists():
                    project_root = parent
                    break

            # Try default locations in order of preference
            for default_path in [
                Path("synplan_data"),  # Relative to current working directory
                project_root / "synplan_data" if project_root else None,  # Project root
                Path.cwd() / "synplan_data",  # Current working directory
                Path.home() / ".synplan_data",  # User home directory
            ]:
                if default_path is not None and default_path.exists():
                    # Verify it has actual data, not just cache metadata
                    has_bb = (
                        default_path / "building_blocks" / "building_blocks_em_sa_ln.smi"
                    ).exists()
                    has_rules = (default_path / "uspto" / "uspto_reaction_rules.pickle").exists()
                    if has_bb and has_rules:
                        data_folder = default_path
                        break

        if data_folder is None:
            logger.warning(
                "SynPlanner data folder not found. Attempting to download data. "
                "This may take a while on first use."
            )
            try:
                target = (project_root / "synplan_data") if project_root else Path("synplan_data")
                data_folder = target.resolve()
                download_all_data(save_to=data_folder)
            except Exception as exc:
                raise SynPlannerError(
                    f"Failed to download SynPlanner data: {exc}. "
                    "Please ensure SynPlanner data is available or set data_folder parameter."
                ) from exc

        # Load building blocks
        building_blocks_path = data_folder / "building_blocks" / "building_blocks_em_sa_ln.smi"
        if not building_blocks_path.exists():
            # Try alternative locations
            building_blocks_path = data_folder / "building_blocks.smi"
            if not building_blocks_path.exists():
                raise SynPlannerError(
                    f"Building blocks file not found in {data_folder}. "
                    "Please ensure SynPlanner data is properly downloaded."
                )

        try:
            self._building_blocks = load_building_blocks(building_blocks_path, standardize=False)
        except Exception as exc:
            raise SynPlannerError(f"Failed to load building blocks: {exc}") from exc

        # Load reaction rules
        reaction_rules_path = data_folder / "uspto" / "uspto_reaction_rules.pickle"
        if not reaction_rules_path.exists():
            # Try alternative locations
            reaction_rules_path = data_folder / "uspto_reaction_rules.pickle"
            if not reaction_rules_path.exists():
                raise SynPlannerError(
                    f"Reaction rules file not found in {data_folder}. "
                    "Please ensure SynPlanner data is properly downloaded."
                )

        try:
            self._reaction_rules = load_reaction_rules(reaction_rules_path)
        except Exception as exc:
            raise SynPlannerError(f"Failed to load reaction rules: {exc}") from exc

        # Load policy network
        ranking_policy_network = data_folder / "uspto" / "weights" / "ranking_policy_network.ckpt"
        if not ranking_policy_network.exists():
            # Try alternative locations
            ranking_policy_network = data_folder / "ranking_policy_network.ckpt"
            if not ranking_policy_network.exists():
                raise SynPlannerError(
                    f"Policy network weights not found in {data_folder}. "
                    "Please ensure SynPlanner data is properly downloaded."
                )

        try:
            policy_config = PolicyNetworkConfig(weights_path=str(ranking_policy_network))
            self._policy_network = PolicyNetworkFunction(policy_config=policy_config)
        except Exception as exc:
            raise SynPlannerError(f"Failed to load policy network: {exc}") from exc

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def identify_input(self, query: str, llm_smiles_guess: Optional[str] = None) -> Dict[str, Any]:
        """Return canonical information about the provided identifier."""

        if not isinstance(query, str):
            raise SynPlannerError("Input must be provided as a string containing SMILES or a name")

        cleaned = query.strip()
        if not cleaned:
            raise SynPlannerError("Input must not be blank")

        smiles_std = standardize_smiles(cleaned)
        if smiles_std is not None:
            return {
                "source": "smiles",
                "query": query,
                "smiles": smiles_std,
            }

        smiles = self.convert_name_to_smiles(cleaned, llm_smiles_guess=llm_smiles_guess)
        smiles_std = standardize_smiles(smiles)
        if smiles_std is None:
            raise SynPlannerError(f"Converted SMILES '{smiles}' from name '{cleaned}' is invalid")
        return {
            "source": "name",
            "query": query,
            "smiles": smiles_std,
        }

    def convert_name_to_smiles(self, name: str, *, llm_smiles_guess: Optional[str] = None) -> str:
        """Convert a molecule name to canonical SMILES using multiple strategies.

        If PubChem lookup fails and an LLM SMILES guess is available, raises
        UserConfirmationRequiredError with the SMILES and image for user confirmation.
        """

        if not isinstance(name, str):
            raise SynPlannerError("Molecule name must be provided as a string")

        cleaned = name.strip()
        if not cleaned:
            raise SynPlannerError("Molecule name must not be blank")

        canonical_llm_guess: Optional[str] = None
        if llm_smiles_guess:
            try:
                canonical_llm_guess = self.mol_to_smiles(self.smiles_to_mol(llm_smiles_guess))
            except (InvalidSMILESError, SynPlannerError) as exc:
                logger.warning(
                    "SMILES can not be pre-processed. Ignoring LLM SMILES guess '%s': %s",
                    llm_smiles_guess,
                    exc,
                )

        pubchem_smiles = self._query_pubchem_smiles(cleaned, canonical_llm_guess)
        if pubchem_smiles:
            return pubchem_smiles

        # If PubChem search failed, check if we have an LLM guess
        if canonical_llm_guess:
            # Generate image for the LLM guess
            try:
                image_data = smiles_to_png_bytes(canonical_llm_guess)
                raise UserConfirmationRequiredError(
                    f"PubChem lookup failed for '{cleaned}'. Please confirm if this is the correct molecule.",
                    smiles=canonical_llm_guess,
                    image_data=image_data,
                    molecule_name=cleaned,
                )
            except ValueError as exc:
                # Invalid SMILES in LLM guess
                raise SynPlannerError(
                    f"PubChem lookup failed for '{cleaned}' and LLM SMILES guess is invalid: {exc}"
                ) from exc

        # No LLM guess available and PubChem failed
        raise SynPlannerError(
            f"Could not resolve molecule name '{cleaned}' to SMILES. "
            "PubChem lookup failed and no valid LLM SMILES guess was provided."
        )

    def _canonicalize_smiles(self, smiles: str) -> str:
        """Convert SMILES to RDKit mol and back to canonical SMILES.

        Args:
            smiles: SMILES string to canonicalize

        Returns:
            Canonical SMILES string

        Raises:
            SynPlannerError: If SMILES is invalid
        """
        try:
            mol = self.smiles_to_mol(smiles)
            return self.mol_to_smiles(mol)
        except (InvalidSMILESError, Exception) as exc:
            raise SynPlannerError(f"Failed to canonicalize SMILES '{smiles}': {exc}") from exc

    def _query_pubchem_smiles(self, name: str, canonical_llm_guess: Optional[str]) -> Optional[str]:
        """Query PubChem for SMILES by name or SMILES.

        If PubChem returns a SMILES, it is converted to RDKit mol and back to
        canonical SMILES before returning.

        Args:
            name: Molecule name to search
            canonical_llm_guess: Optional canonical SMILES from LLM to use as search query

        Returns:
            Canonical SMILES from PubChem if found, None otherwise
        """
        try:
            from pubchempy import get_compounds  # type: ignore import
        except ImportError:
            logger.debug("PubChemPy not installed; skipping PubChem verification")
            return None

        queries: List[tuple[str, str]] = []
        if canonical_llm_guess:
            queries.append(("smiles", canonical_llm_guess))
        queries.append(("name", name))

        seen: set[tuple[str, str]] = set()

        for namespace, value in queries:
            if not value or (namespace, value) in seen:
                continue

            seen.add((namespace, value))

            try:
                compounds = get_compounds(value, namespace=namespace)
            except Exception as exc:  # pragma: no cover - network or API error
                logger.warning("PubChem lookup for %s '%s' failed: %s", namespace, value, exc)
                continue

            if not compounds:
                continue

            for compound in compounds:
                candidate = getattr(compound, "connectivity_smiles", None) or getattr(
                    compound, "isomeric_smiles", None
                )
                if not candidate:
                    continue

                try:
                    # Convert PubChem SMILES to mol and back to canonical SMILES
                    return self._canonicalize_smiles(candidate)
                except SynPlannerError:
                    logger.warning(
                        "PubChem returned an unparsable SMILES '%s' for '%s'",
                        candidate,
                        value,
                    )
                    continue

        return None

    # ------------------------------------------------------------------
    # Planning and formatting
    # ------------------------------------------------------------------
    def plan_synthesis(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        llm_smiles_guess: Optional[str] = None,
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the SynPlanner retrosynthesis engine for the given query.

        Args:
            query: SMILES string or molecule name
            top_k: Number of top routes to return
            llm_smiles_guess: Optional SMILES guess from LLM
            agent: Optional agent instance for storing PNG paths in session state
            session_state: Optional injected session state shared by the team
        """

        info = self.identify_input(query, llm_smiles_guess=llm_smiles_guess)
        smiles = info["smiles"]

        # Load SynPlanner components if not already loaded
        self._load_synplanner_components()

        request_top_k = top_k if top_k is not None else self.default_top_k
        attempts: List[Dict[str, Any]] = []
        tree = None
        raw_routes: List[Any] = []
        successful_attempt: Optional[str] = None

        profiles = self._build_search_profiles()
        for profile in profiles:
            try:
                attempt_tree = self._create_and_search_tree(smiles, profile=profile)
                attempt_routes = self._extract_routes_from_tree(attempt_tree, request_top_k)
                attempts.append(self._summarise_attempt(profile, attempt_tree, attempt_routes))
            except SynPlannerError as exc:
                attempts.append(self._summarise_attempt(profile, error=exc))
                logger.warning("SynPlanner attempt '%s' failed: %s", profile.name, exc)
                continue

            if attempt_routes:
                tree = attempt_tree
                raw_routes = attempt_routes
                successful_attempt = profile.name
                break

        routes = self._normalise_routes(raw_routes)

        # Generate visualizations for routes (stored separately to avoid context overflow)
        route_visualizations = []
        if tree is not None and raw_routes:
            route_visualizations = self._generate_route_visualizations(
                tree,
                raw_routes,
                query=info["query"],
                smiles=smiles,
                agent=agent,
                session_state=session_state,
            )

        descriptors = self.get_basic_descriptors(smiles)
        completed_no_route_attempts = [
            attempt for attempt in attempts if attempt.get("stop_reason") == "no_routes"
        ]
        llm_fallback_allowed = (
            not routes and len(attempts) == len(profiles) and bool(completed_no_route_attempts)
        )

        # Store full plan with visualizations for later retrieval
        full_plan = {
            "query": info["query"],
            "source": info["source"],
            "smiles": smiles,
            "top_k": request_top_k,
            "routes": routes,
            "raw": raw_routes,
            "visualizations": route_visualizations,
            "descriptors": descriptors,
            "attempts": attempts,
            "successful_attempt": successful_attempt,
            "llm_fallback_allowed": llm_fallback_allowed,
        }
        self._last_plan = full_plan

        # Return lightweight plan without large visualization data to prevent context overflow
        # Visualizations are still available via get_route_visualizations()
        plan = {
            "query": info["query"],
            "source": info["source"],
            "smiles": smiles,
            "top_k": request_top_k,
            "routes": routes,
            "descriptors": descriptors,
            "visualization_available": len(route_visualizations) > 0,
            "num_visualizations": len(route_visualizations),
            "attempts": attempts,
            "successful_attempt": successful_attempt,
            "llm_fallback_allowed": llm_fallback_allowed,
        }

        report_plan = self._store_report_ready_plan(
            agent, session_state, plan, route_visualizations
        )
        plan["synthesis_report_data"] = report_plan
        full_plan["synthesis_report_data"] = report_plan

        return plan

    def _build_search_profiles(self) -> List[_SearchProfile]:
        """Return ordered SynPlanner attempts from documented defaults to broader retries."""
        base_tree_config = {
            "max_iterations": self.max_iterations,
            "max_tree_size": self.max_tree_size,
            "max_time": self.max_time,
            "max_depth": self.max_depth,
            "search_strategy": "expansion_first",
            "ucb_type": "uct",
            "c_ucb": 0.1,
            "backprop_type": "muzero",
            "init_node_value": 0.5,
            "min_mol_size": self.min_mol_size,
            "epsilon": 0.0,
            "silent": True,
        }
        base_policy_config = {
            "top_rules": self.top_rules,
            "rule_prob_threshold": self.rule_prob_threshold,
        }

        def profile(
            name: str,
            tree_updates: Optional[Dict[str, Any]] = None,
            policy_updates: Optional[Dict[str, Any]] = None,
        ) -> _SearchProfile:
            tree_config = {**base_tree_config, **(tree_updates or {})}
            policy_config = {**base_policy_config, **(policy_updates or {})}
            return _SearchProfile(name=name, tree_config=tree_config, policy_config=policy_config)

        profiles = [
            profile("standard"),
        ]

        if not self.enable_retry_profiles:
            return profiles

        longer_iterations = max(self.max_iterations * 3, 300)
        deeper_iterations = max(self.max_iterations * 5, 500)
        broader_top_rules = max(self.top_rules * 2, 100)

        profiles.extend(
            [
                profile(
                    "longer_search",
                    {
                        "max_iterations": longer_iterations,
                        "max_time": max(self.max_time * 2, 240),
                    },
                ),
                profile(
                    "deeper_search",
                    {
                        "max_iterations": deeper_iterations,
                        "max_time": max(self.max_time * 3, 360),
                        "max_depth": max(self.max_depth + 3, 12),
                    },
                ),
                profile(
                    "broader_expansion",
                    {
                        "max_iterations": deeper_iterations,
                        "max_time": max(self.max_time * 3, 360),
                        "max_depth": max(self.max_depth + 3, 12),
                    },
                    {"top_rules": broader_top_rules, "rule_prob_threshold": 0.0},
                ),
                profile(
                    "exploratory_uct",
                    {
                        "max_iterations": max(self.max_iterations * 8, 800),
                        "max_time": max(self.max_time * 4, 480),
                        "max_depth": max(self.max_depth + 3, 12),
                        "c_ucb": 0.5,
                        "epsilon": 0.1,
                    },
                    {"top_rules": broader_top_rules, "rule_prob_threshold": 0.0},
                ),
                profile(
                    "evaluation_first",
                    {
                        "max_iterations": max(self.max_iterations * 8, 800),
                        "max_time": max(self.max_time * 4, 480),
                        "max_depth": max(self.max_depth + 3, 12),
                        "search_strategy": "evaluation_first",
                    },
                    {"top_rules": broader_top_rules, "rule_prob_threshold": 0.0},
                ),
            ]
        )

        return profiles

    def _apply_policy_config(self, policy_config: Dict[str, Any]) -> None:
        """Apply per-attempt policy-network parameters to the loaded SynPlanner function."""
        if self._policy_network is None:
            return

        config = getattr(self._policy_network, "config", None)
        if config is None:
            return

        for key, value in policy_config.items():
            setattr(config, key, value)

    def _summarise_attempt(
        self,
        profile: _SearchProfile,
        tree: Optional[Any] = None,
        raw_routes: Optional[List[Any]] = None,
        error: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        route_count = len(raw_routes or [])
        summary: Dict[str, Any] = {
            "profile": profile.name,
            "route_count": route_count,
            "stop_reason": "routes_found" if route_count else "no_routes",
            "parameters": {
                "tree": dict(profile.tree_config),
                "policy": dict(profile.policy_config),
            },
        }

        if tree is not None:
            summary.update(
                {
                    "iterations": getattr(tree, "curr_iteration", None),
                    "tree_size": self._safe_tree_size(tree),
                    "search_time": self._safe_search_time(tree),
                    "max_depth_reached": self._safe_max_depth(tree),
                }
            )

        if error is not None:
            summary["stop_reason"] = "error"
            summary["error"] = str(error)

        return summary

    def _store_report_ready_plan(
        self,
        agent: Optional[Agent],
        session_state: Optional[Dict[str, Any]],
        plan: Dict[str, Any],
        visualizations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Persist compact SynPlanner output for the Report Generator agent."""
        report_plan = self._build_report_ready_plan(plan, visualizations)
        self._persist_route_artifacts(report_plan, agent=agent, session_state=session_state)
        plan_path = self._persist_plan_artifact(report_plan, agent=agent, session_state=session_state)
        if plan_path:
            report_plan["plan_path"] = plan_path

        if session_state is not None:
            session_state["synplanner_plan"] = report_plan

        if agent is not None:
            if agent.session_state is None:
                agent.session_state = {}
            agent.session_state["synplanner_plan"] = report_plan

        for state in update_state_targets(agent, session_state):
            route_ids = []
            target_compound_id = self._existing_session_compound_id(
                state,
                report_plan.get("smiles"),
            )
            if target_compound_id:
                existing = self._session_compound_record(state, target_compound_id) or {}
                related = dict(existing.get("related") or {})
                related["synplanner_query"] = report_plan.get("query")
                target_record = update_session_object(
                    state,
                    target_compound_id,
                    {
                        "source": report_plan.get("source"),
                        "related": related,
                    },
                    set_current=True,
                    current_role="compound",
                )
                target_compound_id = target_record["id"]
            else:
                target_compound_id = register_session_object(
                    state,
                    "compound",
                    {
                        "smiles": report_plan.get("smiles"),
                        "source": report_plan.get("source"),
                        "related": {"synplanner_query": report_plan.get("query")},
                    },
                    label="SynPlanner target compound",
                    source_agent=getattr(agent, "name", None),
                    source_tool="plan_synthesis",
                    set_current=True,
                )
            for idx, route in enumerate(report_plan.get("routes", []), start=1):
                route_id = register_session_object(
                    state,
                    "route",
                    {
                        "target_smiles": report_plan.get("smiles"),
                        "target_compound_id": target_compound_id,
                        "route_index": idx,
                        "score": route.get("score") if isinstance(route, dict) else None,
                        "steps": route.get("steps", []) if isinstance(route, dict) else [],
                        "visualizations": report_plan.get("visualizations", []),
                    },
                    label=f"SynPlanner route {idx}",
                    source_agent=getattr(agent, "name", None),
                    source_tool="plan_synthesis",
                    set_current=idx == 1,
                )
                route_ids.append(route_id)
            if route_ids:
                try:
                    update_session_object(
                        state,
                        target_compound_id,
                        {"related_route_ids": route_ids},
                    )
                except KeyError:
                    pass
        return report_plan

    def _retrosynthesis_rel_path(
        self,
        query: Optional[str],
        smiles: Optional[str],
        *parts: str,
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        state = _session_state_for_outputs(agent, session_state)
        return operation_rel_path(
            OutputOperation.RETROSYNTHESIS,
            "targets",
            _target_slug(query, smiles),
            *parts,
            session_state=state,
            workflow_slug="retrosynthesis",
        )

    def _persist_plan_artifact(
        self,
        report_plan: Dict[str, Any],
        *,
        agent: Optional[Agent],
        session_state: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        rel_path = self._retrosynthesis_rel_path(
            report_plan.get("query"),
            report_plan.get("smiles"),
            "plan.json",
            agent=agent,
            session_state=session_state,
        )
        try:
            payload = {key: value for key, value in report_plan.items() if key != "plan_path"}
            with S3.open(rel_path, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            return S3.path(rel_path)
        except Exception as exc:
            logger.warning("Failed to persist SynPlanner plan artifact: %s", exc)
            return None

    def _persist_route_artifacts(
        self,
        report_plan: Dict[str, Any],
        *,
        agent: Optional[Agent],
        session_state: Optional[Dict[str, Any]],
    ) -> None:
        routes = report_plan.get("routes", [])
        if not isinstance(routes, list):
            return
        for idx, route in enumerate(routes, start=1):
            if not isinstance(route, dict):
                continue
            rel_path = self._retrosynthesis_rel_path(
                report_plan.get("query"),
                report_plan.get("smiles"),
                "routes",
                f"route_{idx:03d}.json",
                agent=agent,
                session_state=session_state,
            )
            try:
                with S3.open(rel_path, "w") as handle:
                    json.dump(route, handle, indent=2, sort_keys=True, default=str)
                route["route_json_path"] = S3.path(rel_path)
            except Exception as exc:
                logger.warning("Failed to persist SynPlanner route %d artifact: %s", idx, exc)

    @staticmethod
    def _existing_session_compound_id(
        session_state: Dict[str, Any],
        smiles: Optional[str],
    ) -> Optional[str]:
        if not smiles:
            return None
        compounds = list_session_objects(session_state, "compound")
        generated_matches = [
            compound
            for compound in compounds
            if compound.get("smiles") == smiles and compound.get("origin_type") == "generated"
        ]
        if generated_matches:
            return generated_matches[-1].get("id")
        for compound in reversed(compounds):
            if compound.get("smiles") == smiles:
                return compound.get("id")
        return None

    @staticmethod
    def _session_compound_record(
        session_state: Dict[str, Any],
        compound_id: str,
    ) -> Optional[Dict[str, Any]]:
        for compound in list_session_objects(session_state, "compound"):
            if compound.get("id") == compound_id:
                return compound
        return None

    def _build_report_ready_plan(
        self,
        plan: Dict[str, Any],
        visualizations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build the compact SynPlanner payload used by reports and tool returns."""
        return {
            "query": plan.get("query"),
            "source": plan.get("source"),
            "smiles": plan.get("smiles"),
            "top_k": plan.get("top_k"),
            "routes": plan.get("routes", []),
            "descriptors": plan.get("descriptors", {}),
            "attempts": plan.get("attempts", []),
            "successful_attempt": plan.get("successful_attempt"),
            "llm_fallback_allowed": plan.get("llm_fallback_allowed", False),
            "visualization_available": plan.get("visualization_available", False),
            "num_visualizations": plan.get("num_visualizations", 0),
            "visualizations": self._compact_visualizations(visualizations or []),
        }

    @staticmethod
    def _compact_visualizations(visualizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only report-safe visualization metadata and file paths."""
        compact = []
        for idx, viz in enumerate(visualizations, start=1):
            node_id = viz.get("node_id")
            score = viz.get("score")
            route_index = viz.get("route_index") or idx
            compact.append(
                {
                    "route_index": route_index,
                    "node_id": node_id,
                    "score": score,
                    "png_path": viz.get("png_path"),
                    "svg_path": viz.get("svg_path"),
                    "caption": f"Route from node #{node_id}"
                    + (f" (score: {score:.3f})" if score is not None else ""),
                }
            )
        return compact

    @staticmethod
    def _safe_tree_size(tree: Any) -> Optional[int]:
        try:
            return len(tree)
        except Exception:
            size = getattr(tree, "curr_tree_size", None)
            if isinstance(size, int):
                return size
            return None

    @staticmethod
    def _safe_search_time(tree: Any) -> Optional[float]:
        search_time = getattr(tree, "curr_time", None)
        if search_time is None:
            return None
        try:
            return round(float(search_time), 3)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_max_depth(tree: Any) -> Optional[int]:
        nodes_depth = getattr(tree, "nodes_depth", None)
        if not isinstance(nodes_depth, dict) or not nodes_depth:
            return None
        try:
            return max(nodes_depth.values())
        except ValueError:
            return None

    def _create_and_search_tree(self, smiles: str, profile: Optional[_SearchProfile] = None) -> Any:
        """Create a SynPlanner Tree and run the search."""
        if profile is None:
            profile = self._build_search_profiles()[0]

        try:
            from synplan.chem.utils import mol_from_smiles as synplan_mol_from_smiles
            from synplan.mcts.tree import Tree
            from synplan.utils.config import RolloutEvaluationConfig, TreeConfig
            from synplan.utils.loading import load_evaluation_function
        except ImportError as exc:
            raise SynPlannerError(f"Failed to import SynPlanner Tree components: {exc}") from exc

        # Convert SMILES to molecule object
        try:
            target_molecule = synplan_mol_from_smiles(
                smiles, clean2d=True, standardize=True, clean_stereo=True
            )
        except Exception as exc:
            raise SynPlannerError(f"Failed to parse SMILES '{smiles}': {exc}") from exc

        self._apply_policy_config(profile.policy_config)

        # Create tree configuration
        tree_config = TreeConfig(**profile.tree_config)

        # Create evaluation function (rollout-based)
        eval_config = RolloutEvaluationConfig(
            policy_network=self._policy_network,
            reaction_rules=self._reaction_rules,
            building_blocks=self._building_blocks,
            min_mol_size=profile.tree_config["min_mol_size"],
            max_depth=profile.tree_config["max_depth"],
        )
        evaluation_function = load_evaluation_function(eval_config)

        # Create and search the tree
        try:
            tree = Tree(
                target=target_molecule,
                config=tree_config,
                reaction_rules=self._reaction_rules,
                building_blocks=self._building_blocks,
                expansion_function=self._policy_network,
                evaluation_function=evaluation_function,
            )
            # Run the search by iterating over the tree
            # The Tree class implements __iter__ and __next__ to perform MCTS search
            for solved, _node_id in tree:
                if solved:
                    break
        except StopIteration:
            # StopIteration is raised when search completes (max iterations, time, or tree size reached)
            pass
        except Exception as exc:
            raise SynPlannerError(f"SynPlanner tree search failed: {exc}") from exc

        return tree

    def _extract_routes_from_tree(self, tree: Any, top_k: int) -> List[Any]:
        """Extract routes from the Tree's winning_nodes."""
        routes = []
        winning_nodes = getattr(tree, "winning_nodes", [])
        if not winning_nodes:
            return routes

        # Get top_k routes
        for node_id in winning_nodes[:top_k]:
            try:
                score = tree.route_score(node_id) if hasattr(tree, "route_score") else None
                route_data = {
                    "node_id": node_id,
                    "score": score,
                    "tree": tree,  # Keep reference to tree for route extraction
                }
                routes.append(route_data)
            except Exception as exc:
                logger.warning(f"Failed to extract route for node {node_id}: {exc}")
                continue

        return routes

    def _normalise_routes(self, routes: Any) -> List[Dict[str, Any]]:
        """Normalize routes extracted from SynPlanner Tree."""
        if routes is None:
            return []

        if not isinstance(routes, list):
            return []

        normalised: List[Dict[str, Any]] = []
        for idx, route_data in enumerate(routes):
            if not isinstance(route_data, dict):
                continue

            tree = route_data.get("tree")
            node_id = route_data.get("node_id")
            score = route_data.get("score")

            if tree is None or node_id is None:
                continue

            # Extract route steps from the tree
            steps = self._extract_route_steps_from_tree(tree, node_id)
            normalised_steps = [step.as_dict() for step in self._normalise_steps(steps)]

            normalised.append(
                {
                    "index": idx,
                    "score": score,
                    "steps": normalised_steps,
                    "num_steps": len(normalised_steps),
                }
            )

        return normalised

    def _extract_route_steps_from_tree(self, tree: Any, node_id: Any) -> List[Any]:
        """Extract reaction steps from a Tree route starting at node_id.

        Uses the Tree.route_to_node() method to get the sequence of nodes,
        then extracts reaction information from each node.
        """
        steps = []
        try:
            # Use the Tree's route_to_node method to get the path
            if hasattr(tree, "route_to_node"):
                route_nodes = tree.route_to_node(node_id)

                # Extract reaction information from consecutive node pairs
                for before_node, after_node in zip(route_nodes, route_nodes[1:], strict=False):
                    try:
                        # Extract reactants (from before node)
                        reactants = []
                        if hasattr(before_node, "curr_precursor"):
                            reactant_mol = before_node.curr_precursor.molecule
                            reactants.append(str(reactant_mol))

                        # Extract products (from after node's new precursors)
                        products = []
                        if hasattr(after_node, "new_precursors"):
                            for precursor in after_node.new_precursors:
                                if hasattr(precursor, "molecule"):
                                    products.append(str(precursor.molecule))

                        # Create a step dictionary
                        step = {
                            "reactants": reactants,
                            "products": products,
                            "description": f"Reaction step: {', '.join(reactants)} -> {', '.join(products)}",
                        }
                        steps.append(step)
                    except Exception as exc:
                        logger.debug(f"Failed to extract step from node pair: {exc}")
                        continue
            else:
                # Fallback: try to access nodes directly
                if hasattr(tree, "nodes") and node_id in tree.nodes:
                    node = tree.nodes[node_id]
                    # Try to extract reaction information from the node
                    if hasattr(node, "reaction"):
                        steps.append(node.reaction)
                    elif hasattr(node, "curr_precursor"):
                        # Create a basic step from available information
                        mol = node.curr_precursor.molecule
                        step = {
                            "reactants": [str(mol)],
                            "products": [],
                            "description": f"Precursor: {str(mol)}",
                        }
                        steps.append(step)
        except Exception as exc:
            logger.warning(f"Failed to extract route steps from tree for node {node_id}: {exc}")

        return steps

    def _normalise_steps(self, steps: Iterable[Any]) -> List[_NormalisedStep]:
        normalised: List[_NormalisedStep] = []

        for idx, step in enumerate(steps, start=1):
            if isinstance(step, dict):
                description = (
                    step.get("description") or step.get("summary") or "SynPlanner reaction step"
                )
                reactants = self._ensure_sequence(step.get("reactants") or step.get("precursors"))
                products = self._ensure_sequence(step.get("products") or step.get("targets"))
                reagents = self._ensure_sequence(step.get("reagents") or step.get("conditions"))
            else:
                description = (
                    getattr(step, "description", None)
                    or getattr(step, "summary", None)
                    or "SynPlanner reaction step"
                )
                reactants = self._ensure_sequence(
                    getattr(step, "reactants", None) or getattr(step, "precursors", None)
                )
                products = self._ensure_sequence(
                    getattr(step, "products", None) or getattr(step, "targets", None)
                )
                reagents = self._ensure_sequence(
                    getattr(step, "reagents", None) or getattr(step, "conditions", None)
                )

            normalised.append(
                _NormalisedStep(
                    index=idx,
                    description=description,
                    reactants=reactants,
                    products=products,
                    reagents=reagents,
                )
            )

        return normalised

    @staticmethod
    def _strip_masks(svg: str) -> str:
        """Remove mask attributes and definitions from SVG to prevent gray fringes.

        Args:
            svg: SVG content as string

        Returns:
            SVG string with masks removed
        """
        # 1) Remove mask attributes on elements
        svg = re.sub(r'\s+mask="url\(#[-\w]+\)"', "", svg)
        # 2) Remove mask definitions entirely
        svg = re.sub(r"<mask\b[\s\S]*?</mask>", "", svg, flags=re.IGNORECASE)
        return svg

    def _export_crisp(
        self, svg_string: str, output_path: str, k: int = 100, background: str = "white"
    ) -> bool:
        """Convert SVG to PNG with pixel-accurate sizing.

        Args:
            svg_string: SVG content as string
            output_path: Path where PNG should be saved (relative to S3 session)
            k: Pixels per user-unit (default 100, meaning 0.01uu = 1px)
            background: Background color (default "white")

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            import cairosvg
        except ImportError:
            logger.warning("cairosvg not available for PNG conversion")
            return False

        # Remove the masks that cause gray fringes
        svg = self._strip_masks(svg_string)

        # Read viewBox to compute integer output size
        m = re.search(
            r'viewBox="[^"]*?(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"',
            svg,
        )
        if not m:
            raise ValueError("SVG has no viewBox; can't compute pixel-accurate size.")
        vw, vh = float(m.group(3)), float(m.group(4))

        try:
            png_bytes = cairosvg.svg2png(
                bytestring=svg.encode("utf-8"),
                output_width=int(round(vw * k)),
                output_height=int(round(vh * k)),
                background_color=background,
            )
            with S3.open(output_path, "wb") as f:
                f.write(png_bytes)
            return True
        except Exception as exc:
            logger.warning(f"SVG to PNG conversion failed: {exc}")
            return False

    def _convert_svg_to_png(self, svg_string: str, output_path: str) -> bool:
        """Convert SVG string to PNG file.

        Args:
            svg_string: SVG content as string
            output_path: Path where PNG should be saved (relative to S3 session)

        Returns:
            True if conversion successful, False otherwise
        """
        return self._export_crisp(svg_string, output_path)

    def _generate_route_visualizations(
        self,
        tree: Any,
        raw_routes: List[Any],
        *,
        query: Optional[str],
        smiles: Optional[str],
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate SVG visualizations for routes, save both SVG and PNG, and store paths in session state.

        Args:
            tree: The SynPlanner Tree object
            raw_routes: List of route data dictionaries with node_id and tree references
            agent: Optional agent instance for storing paths in session state
            session_state: Optional injected session state shared by the team

        Returns:
            List of dictionaries containing visualization data for each route
        """
        import base64

        visualizations = []
        png_paths = []
        svg_paths = []

        try:
            from synplan.utils.visualisation import get_route_svg
        except ImportError:
            logger.warning("SynPlanner visualization module not available")
            return visualizations

        for route_idx, route_data in enumerate(raw_routes, start=1):
            node_id = route_data.get("node_id")
            if node_id is None:
                continue

            try:
                # Generate SVG for the route
                svg_string = get_route_svg(tree, node_id)

                if svg_string:
                    # Convert SVG to base64 data URL for display in UI
                    svg_bytes = svg_string.encode("utf-8")
                    svg_base64 = base64.b64encode(svg_bytes).decode("utf-8")
                    data_url = f"data:image/svg+xml;base64,{svg_base64}"

                    score = route_data.get("score")
                    viz_data = {
                        "route_index": route_idx,
                        "node_id": node_id,
                        "score": score,
                        "svg": svg_string,
                        "svg_data_url": data_url,
                    }

                    # Save SVG to S3
                    svg_filename = f"route_{route_idx:03d}.svg"
                    svg_path = self._retrosynthesis_rel_path(
                        query,
                        smiles,
                        "routes",
                        svg_filename,
                        agent=agent,
                        session_state=session_state,
                    )
                    try:
                        svg_bytes = svg_string.encode("utf-8")
                        with S3.open(svg_path, "wb") as f:
                            f.write(svg_bytes)
                        svg_s3_path = S3.path(svg_path)
                        viz_data["svg_path"] = svg_s3_path
                        viz_data["svg_filename"] = svg_filename
                        svg_paths.append(svg_s3_path)
                        logger.info(f"Saved SVG visualization to {svg_s3_path}")
                    except Exception as exc:
                        logger.warning(f"Failed to save SVG for route node {node_id}: {exc}")

                    # Convert SVG to PNG and save to S3
                    png_filename = f"route_{route_idx:03d}.png"
                    png_path = self._retrosynthesis_rel_path(
                        query,
                        smiles,
                        "routes",
                        png_filename,
                        agent=agent,
                        session_state=session_state,
                    )

                    if self._convert_svg_to_png(svg_string, png_path):
                        # Get the full S3 path for storage in session state
                        png_s3_path = S3.path(png_path)
                        viz_data["png_path"] = png_s3_path
                        viz_data["png_filename"] = png_filename
                        png_paths.append(png_s3_path)
                        logger.info(f"Saved PNG visualization to {png_s3_path}")
                    else:
                        logger.warning(f"Failed to convert SVG to PNG for route node {node_id}")

                    visualizations.append(viz_data)
                else:
                    logger.debug(f"No SVG generated for route node {node_id}")
            except Exception as exc:
                logger.warning(f"Failed to generate visualization for route node {node_id}: {exc}")
                continue

        # Store paths in both possible state carriers. Agno propagates the injected
        # session_state between team members; direct agent calls use agent.session_state.
        state_targets = []
        if session_state is not None:
            state_targets.append(session_state)
        if agent is not None:
            if agent.session_state is None:
                agent.session_state = {}
            if agent.session_state is not session_state:
                state_targets.append(agent.session_state)

        for state in state_targets:
            if png_paths:
                state["synplanner_route_png_paths"] = png_paths
            if svg_paths:
                state["synplanner_route_svg_paths"] = svg_paths

        if png_paths:
            logger.info(f"Stored {len(png_paths)} PNG paths in session state")
        if svg_paths:
            logger.info(f"Stored {len(svg_paths)} SVG paths in session state")

        return visualizations

    @staticmethod
    def _ensure_sequence(value: Any) -> Sequence[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]

    def describe_plan(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        llm_smiles_guess: Optional[str] = None,
    ) -> str:
        """Return a human-readable description of the SynPlanner output."""

        if self._last_plan is None or self._last_plan.get("query") != query:
            plan = self.plan_synthesis(query, top_k=top_k, llm_smiles_guess=llm_smiles_guess)
        else:
            plan = self._last_plan

        if not plan["routes"]:
            attempts = plan.get("attempts", [])
            fallback_note = (
                " LLM fallback is allowed, but any route should be clearly marked as not "
                "SynPlanner-validated."
                if plan.get("llm_fallback_allowed")
                else ""
            )
            return (
                f"SynPlanner did not return any retrosynthetic routes for {plan['smiles']} "
                f"after {len(attempts)} search attempt(s).{fallback_note}"
            )

        lines = [
            f"Retrosynthetic proposal for {plan['query']} ({plan['smiles']}):",
            "",
        ]
        successful_attempt = plan.get("successful_attempt")
        attempts = plan.get("attempts", [])
        if successful_attempt:
            lines.append(
                f"SynPlanner found this route with the '{successful_attempt}' profile "
                f"after {len(attempts)} search attempt(s)."
            )
            lines.append("")

        best_route = plan["routes"][0]
        score = best_route.get("score")
        if score is not None:
            lines.append(f"Best route score: {score}")
        lines.append(f"Number of steps: {best_route['num_steps']}")
        lines.append("")

        for step in best_route["steps"]:
            reagents = f" | Reagents: {', '.join(step['reagents'])}" if step["reagents"] else ""
            lines.append(
                f"Step {step['index']}: {step['description']} (Reactants: {', '.join(step['reactants']) or 'n/a'} -> "
                f"Products: {', '.join(step['products']) or 'n/a'}{reagents})"
            )

        lines.append("")

        return "\n".join(lines)

    def get_route_visualizations(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        llm_smiles_guess: Optional[str] = None,
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get route visualizations (PNG and SVG image paths) for the synthesis plan.

        Returns a dictionary with PNG and SVG file paths that can be displayed in the UI.
        SVG data is excluded from the response to prevent context overflow.

        Args:
            query: SMILES string or molecule name
            top_k: Number of top routes to return
            llm_smiles_guess: Optional SMILES guess from LLM
            agent: Optional agent instance for storing paths in session state
            session_state: Optional injected session state shared by the team
        """
        if self._last_plan is None or self._last_plan.get("query") != query:
            generated_plan = self.plan_synthesis(
                query,
                top_k=top_k,
                llm_smiles_guess=llm_smiles_guess,
                agent=agent,
                session_state=session_state,
            )
            plan = self._last_plan or generated_plan
        else:
            plan = self._last_plan

        visualizations = plan.get("visualizations", [])

        if not visualizations:
            report_plan = self._store_report_ready_plan(
                agent,
                session_state,
                {
                    **plan,
                    "visualization_available": False,
                    "num_visualizations": 0,
                },
                [],
            )
            return {
                "query": plan["query"],
                "smiles": plan["smiles"],
                "message": "No route visualizations available. No routes were found or visualization generation failed.",
                "visualizations": [],
                "attempts": plan.get("attempts", []),
                "successful_attempt": plan.get("successful_attempt"),
                "llm_fallback_allowed": plan.get("llm_fallback_allowed", False),
                "synthesis_report_data": report_plan,
            }

        # Format visualizations with route information (excluding large SVG/base64 data)
        formatted_viz = []
        for idx, viz in enumerate(visualizations, start=1):
            node_id = viz.get("node_id")
            score = viz.get("score")
            png_path = viz.get("png_path")
            svg_path = viz.get("svg_path")
            route_index = viz.get("route_index") or idx

            formatted_viz.append(
                {
                    "route_index": route_index,
                    "node_id": node_id,
                    "score": score,
                    "png_path": png_path,
                    "svg_path": svg_path,
                    "caption": f"Route from node #{node_id}"
                    + (f" (score: {score:.3f})" if score is not None else ""),
                }
            )

        # Get paths from session state if available
        png_paths_from_session = None
        svg_paths_from_session = None
        if session_state is not None:
            png_paths_from_session = session_state.get("synplanner_route_png_paths")
            svg_paths_from_session = session_state.get("synplanner_route_svg_paths")
        if agent is not None and agent.session_state is not None:
            png_paths_from_session = png_paths_from_session or agent.session_state.get(
                "synplanner_route_png_paths"
            )
            svg_paths_from_session = svg_paths_from_session or agent.session_state.get(
                "synplanner_route_svg_paths"
            )

        report_plan = self._store_report_ready_plan(
            agent,
            session_state,
            {
                **plan,
                "visualization_available": True,
                "num_visualizations": len(formatted_viz),
            },
            formatted_viz,
        )

        return {
            "query": plan["query"],
            "smiles": plan["smiles"],
            "num_routes": len(visualizations),
            "visualizations": formatted_viz,
            "png_paths": png_paths_from_session,
            "svg_paths": svg_paths_from_session,
            "attempts": plan.get("attempts", []),
            "successful_attempt": plan.get("successful_attempt"),
            "llm_fallback_allowed": plan.get("llm_fallback_allowed", False),
            "synthesis_report_data": report_plan,
        }


__all__ = ["SynPlannerToolkit", "SynPlannerError", "UserConfirmationRequiredError"]
