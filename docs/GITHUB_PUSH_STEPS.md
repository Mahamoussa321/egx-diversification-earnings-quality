# How to save this project and push code to GitHub

## 1. Open the project folder

Unzip the package and open the folder in VS Code.

## 2. Confirm Git will ignore data

Run:

```bash
git status --ignored
```

You should see the Excel files under `data/raw/` as ignored files.

## 3. Initialize Git

```bash
git init
git add .
git status
```

Check carefully: the Excel data files should **not** appear in the list of files to be committed.

## 4. First commit

```bash
git commit -m "Initial reproducible code package for EGX diversification paper"
```

## 5. Connect to GitHub

Create a new empty GitHub repository, then run the commands GitHub gives you. They will look like:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

## 6. Keep private files local

The zip includes data locally so you can reproduce the work later. But `.gitignore` prevents GitHub upload of:

```text
data/*
outputs/*
```
