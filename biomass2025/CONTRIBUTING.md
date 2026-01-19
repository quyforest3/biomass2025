# 🤝 Contributing to Biomass Estimation

First off, thank you for considering contributing to Biomass Estimation! It's people like you that make this project better for everyone.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

---

## 📜 Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code:

- **Be respectful**: Treat everyone with respect and consideration
- **Be collaborative**: Work together to achieve the best outcomes
- **Be inclusive**: Welcome and support people of all backgrounds
- **Be professional**: Focus on what is best for the community

---

## 🎯 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title**: Descriptive summary of the issue
- **Steps to reproduce**: Detailed steps to reproduce the behavior
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, package versions
- **Screenshots**: If applicable

**Example:**

```markdown
## Bug: Model training fails with large datasets

**Environment:**
- OS: Windows 10
- Python: 3.9.7
- Streamlit: 1.28.0

**Steps to Reproduce:**
1. Load dataset with >10,000 rows
2. Click "Train Models"
3. Observe error message

**Expected:** Models train successfully
**Actual:** MemoryError raised

**Error Message:**
```
MemoryError: Unable to allocate array with shape...
```
```

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use clear title**: Describe the enhancement
- **Detailed description**: Explain why this enhancement would be useful
- **Examples**: Provide examples of how it would work
- **Alternatives**: Describe alternative solutions you've considered

### 🔧 Pull Requests

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Make your changes**
4. **Test thoroughly**
5. **Commit your changes** (`git commit -m 'Add AmazingFeature'`)
6. **Push to branch** (`git push origin feature/AmazingFeature`)
7. **Open a Pull Request**

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool

### Setup Steps

```bash
# 1. Fork and clone
git clone https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub.git
cd BioVision-Analytics-Hub

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development dependencies
pip install -r requirements-dev.txt  # If available

# 5. Run tests
pytest tests/

# 6. Launch dashboard
streamlit run src/dashboard/app.py
```

---

## 🔀 Pull Request Process

### Before Submitting

1. **Update documentation**: Update README, docstrings, etc.
2. **Add tests**: Ensure new features have tests
3. **Run tests**: All tests must pass
4. **Code formatting**: Follow PEP 8 style guide
5. **Update CHANGELOG**: Add entry for your changes

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added and passing
- [ ] All existing tests passing
- [ ] CHANGELOG.md updated

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots here

## Related Issues
Closes #123
```

---

## 📐 Coding Standards

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these specifics:

#### Formatting

```python
# Good
def calculate_biomass(gedi_data, sentinel_data):
    """
    Calculate above-ground biomass from satellite data.
    
    Args:
        gedi_data (pd.DataFrame): GEDI L4A data
        sentinel_data (pd.DataFrame): Sentinel imagery
        
    Returns:
        pd.DataFrame: Merged data with biomass estimates
    """
    merged = gedi_data.merge(sentinel_data, on='lat_lon')
    return merged
```

#### Naming Conventions

- **Variables**: `snake_case` (e.g., `gedi_data`, `feature_importance`)
- **Functions**: `snake_case` (e.g., `train_model()`, `calculate_rmse()`)
- **Classes**: `PascalCase` (e.g., `BiomassPredictor`, `DataLoader`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`, `DEFAULT_PARAMS`)

#### Docstrings

Use Google-style docstrings:

```python
def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train Random Forest model for biomass prediction.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        n_estimators (int, optional): Number of trees. Defaults to 100.
        
    Returns:
        RandomForestRegressor: Trained model
        
    Raises:
        ValueError: If input data is invalid
        
    Example:
        >>> model = train_random_forest(X_train, y_train, n_estimators=200)
        >>> predictions = model.predict(X_test)
    """
    pass
```

#### Imports

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Local
from src.utils import data_loader
from src.models import evaluator
```

---

## 📝 Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Examples

```bash
# Good commits
feat(models): add XGBoost hyperparameter tuning
fix(dashboard): resolve spatial analysis map rendering
docs(readme): update installation instructions
refactor(preprocessing): optimize data merging pipeline

# Bad commits
update code
fix bug
changes
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

```python
import pytest
from src.models import train_random_forest

def test_random_forest_training():
    """Test Random Forest model training."""
    # Arrange
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    
    # Act
    model = train_random_forest(X_train, y_train)
    
    # Assert
    assert model is not None
    assert hasattr(model, 'predict')
    assert model.n_estimators == 100

def test_random_forest_invalid_input():
    """Test Random Forest with invalid input."""
    with pytest.raises(ValueError):
        train_random_forest(None, None)
```

---

## 📚 Documentation

### Code Comments

```python
# Good: Explain WHY, not WHAT
# Use haversine distance because GEDI footprints are circular
distance = haversine(lat1, lon1, lat2, lon2)

# Bad: Obvious comment
# Calculate distance
distance = haversine(lat1, lon1, lat2, lon2)
```

### README Updates

When adding new features:

1. Update feature list
2. Add usage examples
3. Update project structure if needed
4. Add screenshots if applicable

---

## 🏆 Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **CHANGELOG.md**: Version releases
- **GitHub**: Contributors page

---

## ❓ Questions?

Feel free to:

- **Open an issue**: For questions or discussions
- **Email**: support@biomass-estimation.com
- **Authors**: Nguyen Van Quy and Nguyen Hong Hai

---

## 🙏 Thank You!

Your contributions make Biomass Estimation better for everyone in the environmental data science community!

---

<div align="center">

**Happy Contributing! 🎉**

</div>

