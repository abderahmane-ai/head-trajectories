# Contributing to Developmental Trajectories

Thank you for your interest in contributing! This project is part of independent research, but contributions are welcome.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/abderahmane-ai/head-trajectories.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `python run_tests.py`
6. Commit: `git commit -m "Add: your feature description"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python run_tests.py
```

## Code Standards

### Style
- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Keep functions focused and under 50 lines when possible

### Testing
- Add tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

### Documentation
- Update README.md if adding new features
- Add docstrings with clear parameter descriptions
- Include usage examples for new functionality

## Pull Request Process

1. **Update tests**: Add tests for new functionality
2. **Run full test suite**: `python run_tests.py`
3. **Update documentation**: README, docstrings, etc.
4. **Describe changes**: Clear PR description with motivation
5. **Link issues**: Reference any related issues

## Types of Contributions

### Bug Reports
- Use GitHub Issues
- Include minimal reproducible example
- Specify Python version and OS
- Include error messages and stack traces

### Feature Requests
- Open an issue first to discuss
- Explain the use case
- Consider scientific validity (this is research code)

### Code Contributions
- Bug fixes: always welcome
- New features: discuss in issue first
- Performance improvements: include benchmarks
- Documentation: typos, clarity, examples

### Scientific Contributions
- Alternative scoring functions
- New head type definitions
- Improved classification logic
- Statistical analysis methods

## Research Integrity

This is a scientific research project. Contributions should:
- Maintain reproducibility
- Preserve scientific rigor
- Document methodology changes
- Not alter core experimental design without discussion

## Questions?

Open an issue or reach out via the contact information in README.md.

## Code of Conduct

Be respectful, constructive, and professional. This is an academic research project.
