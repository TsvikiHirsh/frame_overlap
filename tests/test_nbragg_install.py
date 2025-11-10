#!/usr/bin/env python3
"""
Test script to verify nbragg installation and dependencies.

This script checks if all required packages are installed correctly,
especially nbragg and its dependency NCrystal which requires C++ compilation.

Run with: python test_nbragg_install.py
"""

import sys
import traceback

def test_imports():
    """Test importing all required packages."""
    print("Testing package imports...")
    print("-" * 60)

    packages = [
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('skimage', 'scikit-image'),
        ('tqdm', 'tqdm'),
        ('lmfit', 'lmfit'),
        ('ncrystal', 'NCrystal (nbragg dependency)'),
        ('nbragg', 'nbragg'),
    ]

    all_success = True
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {display_name:30s} [OK]")
        except ImportError as e:
            print(f"✗ {display_name:30s} [FAILED]")
            print(f"  Error: {e}")
            all_success = False

    print("-" * 60)
    return all_success

def test_nbragg_functionality():
    """Test basic nbragg functionality."""
    print("\nTesting nbragg functionality...")
    print("-" * 60)

    try:
        import nbragg
        print(f"✓ nbragg version: {nbragg.__version__}")

        # Test creating a simple cross-section
        print("  Testing CrossSection creation...")
        xs = nbragg.CrossSection(iron=nbragg.materials["Fe_sg229_Iron-alpha"])
        print("  ✓ CrossSection created successfully")

        # Test creating a transmission model
        print("  Testing TransmissionModel creation...")
        model = nbragg.TransmissionModel(xs, vary_weights=False)
        print("  ✓ TransmissionModel created successfully")

        print("-" * 60)
        return True

    except Exception as e:
        print(f"✗ nbragg functionality test failed")
        print(f"  Error: {e}")
        traceback.print_exc()
        print("-" * 60)
        return False

def test_ncrystal():
    """Test NCrystal directly."""
    print("\nTesting NCrystal (nbragg dependency)...")
    print("-" * 60)

    try:
        import NCrystal
        print(f"✓ NCrystal version: {NCrystal.__version__}")

        # Test loading a material
        print("  Testing material loading...")
        mat = NCrystal.load('Al_sg225.ncmat')
        print(f"  ✓ Material loaded: {mat.info.displayLabel()}")

        print("-" * 60)
        return True

    except Exception as e:
        print(f"✗ NCrystal test failed")
        print(f"  Error: {e}")
        traceback.print_exc()
        print("-" * 60)
        return False

def test_frame_overlap_import():
    """Test importing frame_overlap package with Analysis."""
    print("\nTesting frame_overlap package import...")
    print("-" * 60)

    try:
        # Add src to path
        sys.path.insert(0, 'src')

        from frame_overlap import Data, Reconstruct, Workflow, Analysis
        print("✓ Successfully imported: Data, Reconstruct, Workflow, Analysis")

        # Check if Analysis is working
        print("  Testing Analysis instantiation...")
        analysis = Analysis(xs='iron', vary_background=True)
        print("  ✓ Analysis instantiated successfully")

        print("-" * 60)
        return True

    except Exception as e:
        print(f"✗ frame_overlap import failed")
        print(f"  Error: {e}")
        traceback.print_exc()
        print("-" * 60)
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("nbragg Installation Test Suite")
    print("=" * 60)
    print()

    results = []

    # Test 1: Import all packages
    results.append(("Package Imports", test_imports()))

    # Test 2: NCrystal functionality
    results.append(("NCrystal", test_ncrystal()))

    # Test 3: nbragg functionality
    results.append(("nbragg Functionality", test_nbragg_functionality()))

    # Test 4: frame_overlap import
    results.append(("frame_overlap Import", test_frame_overlap_import()))

    # Summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name:30s} [{status}]")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! nbragg is correctly installed.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nDebugging tips:")
        print("1. Ensure build-essential, cmake, g++, python3-dev are installed")
        print("2. Try: pip install --upgrade nbragg")
        print("3. Check that NCrystal compiled correctly")
        print("4. For Streamlit Cloud, ensure packages.txt contains:")
        print("   build-essential")
        print("   cmake")
        print("   g++")
        print("   python3-dev")
        return 1

if __name__ == '__main__':
    sys.exit(main())
