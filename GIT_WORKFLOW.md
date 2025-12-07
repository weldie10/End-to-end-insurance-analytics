# Git Workflow Guide

This document outlines the git workflow and best practices for this project.

## Branch Strategy

### Main Branches
- `main`: Production-ready code, always stable
- `task-1-*`: Branches for Task 1 (EDA & Statistics)
- `task-2-*`: Branches for Task 2 (DVC setup)

### Branch Naming Convention
- `task-{number}-{feature}`: e.g., `task-1-eda`, `task-2-dvc-init`
- Use kebab-case
- Be descriptive and specific

## Commit Message Guidelines

Follow conventional commit format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `dvc`: DVC-related changes

### Examples
```bash
feat(data): add DataLoader class for data preprocessing
fix(eda): handle missing values in loss ratio calculation
docs(readme): update installation instructions
dvc(config): initialize DVC and configure remote storage
test(data): add unit tests for DataLoader
```

## Workflow Steps

### 1. Create Feature Branch
```bash
git checkout main
git pull origin main
git checkout -b task-1-eda-analysis
```

### 2. Make Small, Focused Commits
```bash
# Work on feature
git add src/alphacare/eda/eda_analyzer.py
git commit -m "feat(eda): add loss ratio calculation method"

# Continue working
git add src/alphacare/eda/eda_analyzer.py
git commit -m "feat(eda): add outlier detection functionality"
```

### 3. Push Branch
```bash
git push origin task-1-eda-analysis
```

### 4. Create Pull Request
- Use descriptive title
- Fill out PR template
- Request review
- Link to related task/issue

### 5. Merge via PR
- Review feedback addressed
- All checks pass
- Squash and merge (or merge commit)

## Best Practices

### ✅ Do
- Make small, frequent commits (1 logical change per commit)
- Write clear, descriptive commit messages
- Create branches for each feature/task
- Use pull requests for code review
- Keep commits focused on one thing
- Test before committing

### ❌ Don't
- Commit large files (use DVC)
- Make huge commits with many unrelated changes
- Commit directly to main
- Use vague commit messages like "fix" or "update"
- Force push to shared branches

## Example Workflow

### Task 1: EDA Analysis

```bash
# 1. Create branch
git checkout main
git checkout -b task-1-eda-analysis

# 2. Add EDA class structure
git add src/alphacare/eda/
git commit -m "feat(eda): create EDAAnalyzer class structure"

# 3. Add descriptive statistics
git add src/alphacare/eda/eda_analyzer.py
git commit -m "feat(eda): implement descriptive statistics calculation"

# 4. Add loss ratio analysis
git add src/alphacare/eda/eda_analyzer.py
git commit -m "feat(eda): add loss ratio calculation by groups"

# 5. Add outlier detection
git add src/alphacare/eda/eda_analyzer.py
git commit -m "feat(eda): implement outlier detection methods"

# 6. Push and create PR
git push origin task-1-eda-analysis
```

### Task 2: DVC Setup

```bash
# 1. Create branch
git checkout main
git checkout -b task-2-dvc-init

# 2. Initialize DVC
dvc init
git add .dvc/
git commit -m "dvc(init): initialize DVC repository"

# 3. Configure remote
dvc remote add -d localstorage /path/to/storage
git add .dvc/config
git commit -m "dvc(config): add local storage remote"

# 4. Add data file
dvc add data/raw/MachineLearningRating_v3.txt
git add data/raw/MachineLearningRating_v3.txt.dvc .gitignore
git commit -m "dvc(data): track data file with DVC"

# 5. Push
git push origin task-2-dvc-init
```

## Commit Frequency

Aim for:
- **Small commits**: 1-3 files changed per commit
- **Frequent commits**: Every 30-60 minutes of work
- **Logical grouping**: Related changes together
- **Clear messages**: Anyone can understand what changed

## Review Process

1. **Self-review**: Check your changes before committing
2. **Push frequently**: Don't wait until everything is done
3. **Create PR early**: Mark as "Draft" if not ready
4. **Address feedback**: Make changes in new commits
5. **Keep PR focused**: One feature/task per PR

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Best Practices](https://github.com/git/git/blob/master/Documentation/SubmittingPatches)
- [DVC Documentation](https://dvc.org/doc)

