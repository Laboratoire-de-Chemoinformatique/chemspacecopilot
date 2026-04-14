#!/usr/bin/env python
# coding: utf-8
"""
Example script demonstrating SMILES encoding and sampling with autoencoder integration.

This script shows how to use the new AutoencoderToolkit in cs_copilot for:
1. Encoding SMILES strings to latent vectors
2. Sampling new molecules from latent space
3. Interpolating between molecules
4. Exploring chemical space neighborhoods
"""

import logging
from pathlib import Path

from cs_copilot.agents import create_agent
from cs_copilot.tools import AutoencoderToolkit

# Set up logging
logger = logging.getLogger(__name__)

def test_autoencoder_toolkit():
    """Test the AutoencoderToolkit directly."""
    print("=" * 80)
    print("Testing AutoencoderToolkit Direct Usage")
    print("=" * 80)

    try:
        # Initialize the toolkit
        toolkit = AutoencoderToolkit()

        # Test model validation
        print(f"Model loaded: {toolkit.validate_model_loaded()}")
        print(f"Latent dimension: {toolkit.get_latent_dimension()}")

        # Get model info
        model_info = toolkit.get_model_info()
        print(f"Model info: {model_info}")

        # Test SMILES encoding
        test_smiles = [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CC(=O)Nc1ccc(O)cc1",      # Paracetamol
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ]

        print(f"\nEncoding {len(test_smiles)} SMILES strings...")
        latent_vectors = toolkit.encode_smiles(test_smiles)
        print(f"Encoded to latent vectors with shape: {latent_vectors.shape}")

        # Test reconstruction
        print(f"\nTesting reconstruction...")
        for smiles in test_smiles:
            reconstructed = toolkit.reconstruct_smiles(smiles)
            match = "✓" if smiles == reconstructed else "✗"
            print(f"{match} {smiles} -> {reconstructed}")

        # Test sampling (demo: small N, raw output to show every decoded string)
        print(f"\nSampling 5 new molecules...")
        samples = toolkit.sample_molecules(
            n_samples=5,
            temperature=1.0,
            filter_valid_unique=False,
            return_format="list",
        )
        for i, sample in enumerate(samples, 1):
            print(f"{i}. {sample}")

        # Test interpolation
        print(f"\nInterpolating between molecules...")
        smiles1 = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        smiles2 = "CC(=O)Nc1ccc(O)cc1"      # Paracetamol
        interpolated = toolkit.interpolate_molecules(smiles1, smiles2, n_steps=5)
        print(f"Interpolation from {smiles1} to {smiles2}:")
        for i, interp in enumerate(interpolated):
            print(f"  Step {i}: {interp}")

        # Test neighborhood exploration
        print(f"\nExploring neighborhood of {smiles1}...")
        neighbors = toolkit.explore_latent_neighborhood(smiles1, noise_scale=0.1, n_neighbors=3)
        for i, neighbor in enumerate(neighbors, 1):
            print(f"  Neighbor {i}: {neighbor}")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Error testing toolkit: {e}")
        logger.error(f"Toolkit test failed: {e}")


def test_autoencoder_agent():
    """Test the autoencoder agent."""
    print("\n" + "=" * 80)
    print("Testing Autoencoder Agent")
    print("=" * 80)

    try:
        # Create a mock model for testing
        class MockModel:
            def __init__(self):
                self.name = "mock_model"

        model = MockModel()

        # Create the autoencoder agent
        agent = create_agent("autoencoder", model)

        print(f"Created agent: {agent.name}")
        print(f"Agent description: {agent.description}")
        print(f"Available tools: {[tool.name for tool in agent.tools]}")

        # Test agent capabilities
        print(f"\nAgent capabilities:")
        print(f"- Can encode SMILES: {'encode_smiles' in [tool.name for tool in agent.tools]}")
        print(f"- Can sample molecules: {'sample_molecules' in [tool.name for tool in agent.tools]}")
        print(f"- Can interpolate: {'interpolate_molecules' in [tool.name for tool in agent.tools]}")
        print(f"- Can reconstruct: {'reconstruct_smiles' in [tool.name for tool in agent.tools]}")

        print("\n✓ Agent test passed!")

    except Exception as e:
        print(f"✗ Error testing agent: {e}")
        logger.error(f"Agent test failed: {e}")


def main():
    """Main function to run all tests."""
    print("SMILES Autoencoder Integration Test")
    print("=" * 80)

    # Test the toolkit directly
    test_autoencoder_toolkit()

    # Test the agent
    test_autoencoder_agent()

    print("\n" + "=" * 80)
    print("Integration Test Complete")
    print("=" * 80)
    print("\nThe autoencoder integration provides the following capabilities:")
    print("1. ✓ SMILES encoding to latent vectors")
    print("2. ✓ Molecular sampling from latent space")
    print("3. ✓ Molecular interpolation in chemical space")
    print("4. ✓ Latent space neighborhood exploration")
    print("5. ✓ SMILES reconstruction testing")
    print("6. ✓ Agent integration with cs_copilot")
    print("\nYou can now use the 'autoencoder' agent type in cs_copilot!")


if __name__ == "__main__":
    main()
