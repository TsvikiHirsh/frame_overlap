Installation
============

Requirements
------------

- Python 3.9+
- numpy
- pandas
- matplotlib
- scipy
- scikit-image
- tqdm

Optional dependencies:

- nbragg (for material analysis)
- lmfit (for parameter optimization)
- jupyter (for notebooks)

Using pip
---------

Install the latest release from PyPI:

.. code-block:: bash

   pip install frame-overlap

Install with optional dependencies:

.. code-block:: bash

   # With nbragg support
   pip install frame-overlap[nbragg]

   # With all optional dependencies
   pip install frame-overlap[all]

From Source
-----------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/TsvikiHirsh/frame_overlap.git
   cd frame_overlap
   pip install -e .

With development dependencies:

.. code-block:: bash

   pip install -e .[dev]

Verification
------------

Verify the installation:

.. code-block:: python

   import frame_overlap
   print(frame_overlap.__version__)

   # Test basic functionality
   from frame_overlap import Data, Workflow
   print("✓ Installation successful!")

Docker
------

A Docker image is available (coming soon):

.. code-block:: bash

   docker pull ghcr.io/tsvikihirsh/frame_overlap:latest
   docker run -it --rm ghcr.io/tsvikihirsh/frame_overlap:latest

Troubleshooting
---------------

**ImportError: No module named 'nbragg'**

Install nbragg separately:

.. code-block:: bash

   pip install nbragg

**Permission denied errors**

Use a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install frame-overlap
