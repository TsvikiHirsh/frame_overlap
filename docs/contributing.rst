Contributing
============

We welcome contributions to frame_overlap!

How to Contribute
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature
4. Make your changes
5. Run tests
6. Submit a pull request

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/frame_overlap.git
   cd frame_overlap
   pip install -e .[dev]

Running Tests
-------------

.. code-block:: bash

   pytest

Code Style
----------

We follow PEP 8. Format your code with:

.. code-block:: bash

   black src/frame_overlap
   isort src/frame_overlap

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

Pull Request Guidelines
-----------------------

- Include tests for new features
- Update documentation as needed
- Follow existing code style
- Write clear commit messages
- Reference related issues

Reporting Issues
----------------

Report bugs and request features on `GitHub Issues <https://github.com/TsvikiHirsh/frame_overlap/issues>`_.

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.
