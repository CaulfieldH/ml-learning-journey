# CLAUDE.md - AI Assistant Guide for ML Learning Journey

**Last Updated:** 2025-11-16
**Repository:** ml-learning-journey
**Owner:** CaulfieldH (radiomaximus@gmail.com)

---

## 📋 Table of Contents

1. [Repository Overview](#repository-overview)
2. [Directory Structure](#directory-structure)
3. [Development Environment](#development-environment)
4. [Naming Conventions](#naming-conventions)
5. [Language & Localization](#language--localization)
6. [Current Projects](#current-projects)
7. [Code Patterns & Standards](#code-patterns--standards)
8. [Workflows for AI Assistants](#workflows-for-ai-assistants)
9. [Important Notes](#important-notes)

---

## 🎯 Repository Overview

### Purpose
This repository documents an **18-month learning journey** from industrial automation to Machine Learning Engineering. It serves as both a portfolio and knowledge base for progressive ML skill development.

### Learning Plan
- **Months 1-12:** Skillbox "Machine Learning Engineer" course
- **Months 13-18:** LLM and multi-agent systems specialization
- **Focus:** Industrial applications of ML/AI

### Goals
1. ✅ Become an ML Engineer
2. ✅ Master full AI stack (data science → LLM)
3. ✅ Specialize in industrial applications

### Current Status
- **Week 1:** ✅ Environment setup, first ML project complete
- **Week 2:** 🔄 Python for Data Science (in progress)
- **Skills Progress:**
  - Python for ML: 3→4 📈
  - Classical ML: 0→2 📈
  - Deep Learning: 0→0 ⏳

---

## 📁 Directory Structure

```
ml-learning-journey/
├── .git/                          # Git version control
├── .gitignore                     # Currently empty - needs configuration
├── README.md                      # Main project overview (Russian)
├── requirements.txt               # Python dependencies
│
├── projects/                      # Active ML projects and exercises
│   ├── 01-netology_ml_l8/        # Boston Housing Analysis (ACTIVE)
│   │   ├── .vscode/               # VSCode configuration
│   │   │   ├── launch.json        # Python debugger config
│   │   │   └── settings.json      # Workspace settings
│   │   ├── src/                   # Python source code
│   │   │   ├── main.py            # Basic implementation (86 lines)
│   │   │   └── main_by_cursor.py  # Production version (202 lines)
│   │   ├── data/                  # Data directory (placeholder)
│   │   ├── notebooks/             # Jupyter notebooks (placeholder)
│   │   └── README.md              # Project documentation
│   │
│   ├── 02-industrial-data/        # Planned industrial data analysis
│   │   └── README.md              # Placeholder
│   │
│   └── readme.md                  # Projects index
│
├── notes/                         # Learning materials & documentation
│   ├── readme.md                  # Notes index
│   ├── cheatsheets/               # Quick reference guides
│   │   └── README.md
│   ├── concepts/                  # Conceptual explanations
│   │   └── README.md
│   ├── questions/                 # Q&A and troubleshooting
│   │   ├── README.md
│   │   └── claude_example_a_venv.md  # 50KB environment setup guide
│   └── README.md
│
├── weeks/                         # Weekly learning progress
│   ├── week01-setup-26.05.25/     # Environment setup
│   ├── week02-python-basics-02.06.25/  # Python fundamentals
│   ├── week03-pandas-numpy/       # Data manipulation
│   └── [More weeks to come...]
│
└── resources/                     # External learning materials
    ├── readme.md                  # Resources index
    ├── books/                     # Book references
    ├── courses/                   # Course materials
    └── articles/                  # Article references
```

### Directory Purposes

| Directory | Purpose | Status |
|-----------|---------|--------|
| `projects/` | Working ML projects and course exercises | Active |
| `notes/` | Learning documentation, Q&A, troubleshooting | Active |
| `weeks/` | Week-by-week progress tracking | Active |
| `resources/` | External references and materials | Placeholder |

---

## 🛠️ Development Environment

### Python Environment
- **Python Version:** 3.11+
- **Package Manager:** pip
- **Virtual Environment:** venv (centralized location)
  - **Windows Path:** `C:\Prog\envs\ml_env`
  - **Activation:** See `notes/questions/claude_example_a_venv.md`

### Required Dependencies
```txt
pandas>=1.3.0           # Data manipulation
numpy>=1.20.0           # Numerical computing
scikit-learn>=1.0.0     # Machine learning
matplotlib>=3.3.0       # Visualization
seaborn>=0.11.0         # Statistical plots
```

### IDE Configuration
- **Primary IDE:** Visual Studio Code
- **Required Extensions:**
  - Python (Microsoft)
  - Pylance
  - Python Debugger
  - Jupyter
  - Python Docstring Generator

### VSCode Settings
- Python interpreter: Points to centralized venv
- Debugger configuration: `.vscode/launch.json`
- Workspace settings: `.vscode/settings.json`
- Auto-activation of virtual environment on terminal open

### Setup Instructions
Complete environment setup guide: `notes/questions/claude_example_a_venv.md`

---

## 📐 Naming Conventions

### Directory Naming Patterns

#### Weekly Directories
**Pattern:** `week[NN]-[topic]-[DD.MM.YY]`

**Examples:**
- `week01-setup-26.05.25`
- `week02-python-basics-02.06.25`
- `week03-pandas-numpy`

**Components:**
- `[NN]`: Two-digit week number (01, 02, 03...)
- `[topic]`: Descriptive topic name (kebab-case)
- `[DD.MM.YY]`: Optional start date (European format)

#### Project Directories
**Pattern:** `[NN]-[project_name]`

**Examples:**
- `01-netology_ml_l8`
- `02-industrial-data`

**Components:**
- `[NN]`: Two-digit sequence number for ordering
- `[project_name]`: Descriptive name (snake_case or kebab-case)

### File Naming

| File Type | Pattern | Example |
|-----------|---------|---------|
| Main scripts | `main.py` | `main.py` |
| Script variants | `main_by_[tool].py` | `main_by_cursor.py` |
| Documentation | `README.md` or `readme.md` | ⚠️ Inconsistent casing |
| Config files | `.vscode/[name].json` | `launch.json` |
| Notebooks | `[descriptive_name].ipynb` | (None yet) |

### Code Naming Standards

#### Python Conventions
- **Variables:** `snake_case`
- **Functions:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private methods:** `_leading_underscore`

#### Examples from Codebase
```python
# Functions
def load_data():
def preprocess_data(df, test_size=0.2, random_state=42):
def train_models(X_train, y_train):

# Variables
data_url = "http://lib.stat.cmu.edu/datasets/boston"
trained_models = {}
feature_importance = pd.DataFrame(...)

# Constants
# (None defined yet - would use UPPER_SNAKE_CASE)
```

---

## 🌍 Language & Localization

### Primary Language: Russian (Русский)

**Russian is used for:**
- ✅ All documentation (README.md files)
- ✅ Git commit messages
- ✅ Code comments
- ✅ Learning notes and explanations
- ✅ Project descriptions

**English is used for:**
- ✅ Python code (variables, functions, classes)
- ✅ Technical terms and library names
- ✅ Dataset column names (CRIM, MEDV, etc.)
- ✅ Log messages in code

### Translation Notes for AI Assistants

**Common Terms:**
| Russian | English | Context |
|---------|---------|---------|
| Улучшение структуры репозитория | Repository structure improvement | Git commits |
| практическое задание | Practical assignment | Project type |
| О проекте | About project | Documentation |
| Цели | Goals | Documentation |
| Прогресс | Progress | Tracking |
| Проекты | Projects | Section header |
| Навыки | Skills | Progress tracking |

### When to Use Each Language

**AI Assistants should:**
- Write code comments in Russian to match existing style
- Write commit messages in Russian
- Write documentation/README updates in Russian
- Use English for code identifiers (functions, variables, classes)
- Use English for technical library/framework references

---

## 🔬 Current Projects

### Project 01: Boston Housing Analysis ✅ Active

**Location:** `projects/01-netology_ml_l8/`
**Status:** In progress (from Netology ML Lesson 8)
**Type:** Regression analysis

#### Dataset
- **Name:** Boston Housing Dataset
- **Source:** http://lib.stat.cmu.edu/datasets/boston
- **Records:** 506 samples
- **Features:** 13 input variables + 1 target

**Features:**
- `CRIM`: Per capita crime rate
- `ZN`: Proportion of residential land zoned
- `INDUS`: Proportion of non-retail business
- `CHAS`: Charles River indicator (1 if tract bounds river, 0 otherwise)
- `NOX`: Nitrogen oxide concentration (parts per 10 million)
- `RM`: Average number of rooms per dwelling
- `AGE`: Proportion of owner-occupied units built before 1940
- `DIS`: Weighted distances to employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Property tax rate per $10,000
- `PTRATIO`: Pupil-teacher ratio by town
- `B`: 1000(Bk - 0.63)^2 where Bk is proportion of Black residents
- `LSTAT`: Percentage of lower status population

**Target Variable:**
- `MEDV`: Median value of owner-occupied homes ($1000s)

#### Implementation Files

**1. `src/main.py` (Basic version - 86 lines)**
- Minimal implementation
- Direct translation from notebook style
- Basic linear regression
- Simple evaluation

**2. `src/main_by_cursor.py` (Production version - 202 lines)** ⭐ Recommended
- Proper Python project structure
- Logging configuration
- Argparse CLI interface
- Multiple models (Linear Regression + Random Forest)
- Comprehensive evaluation metrics
- Cross-validation (5-fold)
- Feature importance analysis
- Visualization outputs (PNG files)
- Error handling
- Docstrings for all functions

#### Models Used
1. **Linear Regression** (baseline)
2. **Random Forest Regressor** (100 estimators)

#### Evaluation Metrics
- Mean Squared Error (MSE)
- R² Score
- Cross-validation scores
- Feature importance (Random Forest)
- Predictions vs. Actual plots

#### Output Files
- `feature_importance.png` - Feature importance visualization
- `linear_regression_predictions.png` - Linear model predictions
- `random_forest_predictions.png` - Random Forest predictions

#### Running the Project
```bash
# Navigate to project directory
cd projects/01-netology_ml_l8

# Run basic version
python src/main.py

# Run production version (recommended)
python src/main_by_cursor.py

# With custom parameters
python src/main_by_cursor.py --test-size 0.3 --random-state 123
```

### Project 02: Industrial Data Analysis 📋 Planned

**Location:** `projects/02-industrial-data/`
**Status:** Placeholder only
**Purpose:** Real-world industrial automation data analysis
**Details:** Not yet specified

---

## 💻 Code Patterns & Standards

### Python Code Style

#### General Standards
- ✅ Follow PEP 8
- ✅ UTF-8 encoding declaration: `# -*- coding: utf-8 -*-`
- ✅ Shebang for executables: `#!/usr/bin/env python`
- ✅ Comprehensive docstrings (Google/NumPy style)
- ✅ Type hints (not yet used, but recommended)

#### Logging Pattern
```python
import logging

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use throughout code
logger.info("Data loaded successfully")
logger.warning("Missing values found")
logger.error("Error loading data")
```

#### CLI Arguments Pattern
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data for testing')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
```

#### Function Documentation Pattern
```python
def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the data by splitting into features and target.

    Args:
        df (pd.DataFrame): Input DataFrame
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Implementation
    pass
```

#### Error Handling Pattern
```python
try:
    # Main logic
    df = load_data()
except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    sys.exit(1)
```

### Machine Learning Patterns

#### Standard ML Pipeline
```python
# 1. Load data
df = load_data()

# 2. Preprocess and split
X_train, X_test, y_train, y_test = preprocess_data(df)

# 3. Train models
models = train_models(X_train, y_train)

# 4. Evaluate models
evaluate_models(models, X_train, X_test, y_train, y_test)
```

#### Model Dictionary Pattern
```python
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
```

#### Visualization Pattern
```python
plt.figure(figsize=(10, 6))
# Plotting code
plt.title('Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.savefig('filename.png')
plt.close()  # Important: close to free memory
```

### Project Structure Pattern

For new ML projects, follow this structure:

```
project-name/
├── .vscode/                # IDE configuration
│   ├── launch.json
│   └── settings.json
├── src/                    # Source code
│   ├── main.py
│   └── [additional modules]
├── data/                   # Data files (if local)
│   ├── raw/
│   └── processed/
├── notebooks/              # Jupyter notebooks
│   └── exploration.ipynb
├── tests/                  # Unit tests (future)
│   └── test_main.py
├── outputs/                # Generated files
│   └── *.png
├── README.md               # Project documentation
└── requirements.txt        # Project-specific dependencies
```

---

## 🤖 Workflows for AI Assistants

### When Working on This Repository

#### 1. Understand the Context
- **Language:** All documentation and comments should be in Russian
- **Stage:** Early learning journey, building fundamentals
- **Audience:** The repository owner is learning ML from scratch
- **Tone:** Educational, patient, thorough explanations

#### 2. Before Making Changes

**Check:**
- [ ] Current project status in `README.md`
- [ ] Relevant week directory for context
- [ ] Existing code patterns in similar files
- [ ] Dependencies in `requirements.txt`

**Ask yourself:**
- Is this aligned with the current week's focus?
- Does this match the skill level progression?
- Are there learning opportunities to highlight?

#### 3. When Adding New Projects

**Create structure:**
```bash
projects/
└── NN-project-name/
    ├── .vscode/
    │   ├── launch.json
    │   └── settings.json
    ├── src/
    │   └── main.py
    ├── data/
    ├── notebooks/
    └── README.md
```

**Update tracking:**
- Add entry to `projects/readme.md`
- Update main `README.md` if significant
- Create appropriate week entry if applicable

#### 4. When Writing Code

**Follow these principles:**
- Write production-quality code with proper structure
- Include comprehensive docstrings
- Add logging for visibility into execution
- Include error handling
- Make CLI-friendly with argparse
- Generate visualizations where appropriate
- Comments in Russian, code in English

**Example template:**
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Краткое описание скрипта на русском языке.
Brief description in Russian.
"""

import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description='Описание')
    # Add arguments
    args = parser.parse_args()

    try:
        # Main logic
        pass
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#### 5. When Writing Documentation

**Structure for README.md files:**
```markdown
# Название проекта

## О проекте
[Описание проекта]

## Цели
- Цель 1
- Цель 2

## Структура
[Описание структуры]

## Использование
```bash
# Команды для запуска
```

## Результаты
[Описание результатов]

## Источники
- [Ссылки на материалы]
```

#### 6. Git Commit Messages

**Format:** Russian language, descriptive

**Examples:**
- ✅ "Добавлен проект анализа Boston Housing"
- ✅ "Улучшена структура кода с логированием"
- ✅ "Исправлена ошибка в предобработке данных"
- ❌ "Added Boston Housing project" (English)
- ❌ "Fixed bug" (Too vague)
- ❌ "Update" (Not descriptive)

**Pattern:**
```
Глагол в прошедшем времени + объект изменения + детали

Examples:
- Добавлен [что]
- Обновлен [что]
- Исправлена [что]
- Улучшена [что]
- Удалён [что]
```

#### 7. When Updating This File (CLAUDE.md)

**Keep updated:**
- New projects and their status
- Changed directory structure
- New conventions or patterns
- Dependencies updates
- Progress milestones

**Format consistency:**
- Maintain table of contents
- Use consistent heading levels
- Keep examples up-to-date
- Update "Last Updated" date

---

## ⚠️ Important Notes

### Current Gaps & Limitations

**⚠️ Items to be aware of:**

1. **No .gitignore configuration**
   - File is present but empty
   - Python artifacts, venv, cache files not ignored
   - Recommend configuring for Python projects

2. **Many placeholder directories**
   - Structure exists but content not yet created
   - Don't assume directories contain files
   - Check before referencing

3. **Inconsistent README naming**
   - Mix of `README.md` and `readme.md`
   - Future: standardize to `README.md`

4. **No Jupyter notebooks yet**
   - Directories exist but no `.ipynb` files
   - Project uses scripts instead currently

5. **No test suite**
   - No unit tests or integration tests
   - Consider adding as skills progress

6. **No CI/CD**
   - No automated testing or deployment
   - Manual execution only

7. **Data not stored locally**
   - Boston Housing loaded from remote URL
   - No local data files in repository
   - Consider caching for offline work

8. **Centralized virtual environment**
   - Not in project directory
   - Located at `C:\Prog\envs\ml_env`
   - Requires manual activation

### Security Considerations

**⚠️ Boston Housing Dataset Note:**
The LSTAT and B features in the Boston Housing dataset contain socioeconomic and racial demographic information. Modern ML practice recommends:
- Being aware of potential biases in this historical dataset
- Considering ethical implications when using these features
- This is primarily for educational purposes

### Best Practices for AI Assistants

**DO:**
- ✅ Write all documentation in Russian
- ✅ Use Russian for code comments
- ✅ Follow existing code patterns
- ✅ Include educational explanations
- ✅ Test code before committing
- ✅ Update progress tracking
- ✅ Create visualizations for results
- ✅ Log execution details

**DON'T:**
- ❌ Mix English into Russian documentation
- ❌ Skip docstrings or comments
- ❌ Create overly complex solutions for the current skill level
- ❌ Ignore the learning journey context
- ❌ Leave broken or incomplete code
- ❌ Forget to update README files
- ❌ Commit without testing

### Useful References

**Internal Documentation:**
- Environment setup: `notes/questions/claude_example_a_venv.md`
- Main README: `README.md`
- Project READMEs: `projects/*/README.md`

**External Resources:**
- Boston Housing Dataset: http://lib.stat.cmu.edu/datasets/boston
- scikit-learn docs: https://scikit-learn.org/
- pandas docs: https://pandas.pydata.org/

---

## 📞 Contact & Support

**Repository Owner:** CaulfieldH
**Email:** radiomaximus@gmail.com
**Learning Platform:** Skillbox "Machine Learning Engineer"

---

## 🔄 Changelog for CLAUDE.md

**2025-11-16:**
- Initial creation of CLAUDE.md
- Documented repository structure and conventions
- Added code patterns and workflows
- Included current project status (Boston Housing)

---

**End of AI Assistant Guide**

*This file should be updated as the repository evolves and new patterns emerge.*
