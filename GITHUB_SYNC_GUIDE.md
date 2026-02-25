# Guide to Syncing the ALBuMS Project to GitHub

## Step 1: Initialize Git Repository

```bash
cd /home/lu/streamlit/DRFB

# Initialize Git repository
git init

# Configure Git user info (if not already done)
git config user.name "Your Name"
git config user.email "your_email@example.com"
```

## Step 2: Add Files to Git

```bash
# Check files to be added
git status

# Add all files (.gitignore will automatically exclude unnecessary ones)
git add .

# Check staged files
git status

# Create the first commit
git commit -m "Initial commit: ALBuMS Streamlit application"
```

## Step 3: Create a Repository on GitHub

1. **Visit GitHub**: https://github.com
2. **Log in** to your GitHub account.
3. **Click the "+" button** in the upper right corner → select "New repository".
4. **Enter repository information**:
   - Repository name: `albums-streamlit` (or any name you prefer)
   - Description: `ALBuMS - Advanced Longitudinal Beam Stability Analysis`
   - Choose **Public** or **Private**.
   - **Do not** check "Initialize this repository with a README".
   - Click "Create repository".

## Step 4: Connect Local Repository to GitHub

GitHub will display some commands; use them as follows:

```bash
# Add the remote repository (replace with your GitHub username and repository name)
git remote add origin https://github.com/YourUsername/albums-streamlit.git

# Verify the remote repository
git remote -v

# Push to GitHub (first-time push)
git branch -M main
git push -u origin main
```

## Step 5: Subsequent Updates

For every change hereafter, use the following commands to sync:

```bash
# Check modified files
git status

# Add modified files
git add .

# Commit changes
git commit -m "Describe your changes"

# Push to GitHub
git push
```

---

## Quick Command Script

I have prepared an automated script for you; simply run the following:

```bash
cd /home/lu/streamlit/DRFB
./sync_to_github.sh
```

---

## Important Considerations

### ✅ Files that will be committed:
- All Python source code (`.py` files)
- Configuration files (`requirements.txt`, `Dockerfile`, etc.)
- Documentation (`README.md`, `*.md` files)
- Example files (`examples/` directory)

### ❌ Files that will NOT be committed (already excluded in .gitignore):
- Virtual environment (`.venv/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Downloaded dependencies (`mbtrack2-stable/`, `collective_effects/`)
- Data files (`data/`, `*.h5`, `*.csv`)
- Temporary files (`*.tar.gz`, `.gemini/`)

---

## Frequently Asked Questions

### Q: How do I check which files are about to be committed?
```bash
git status
```

### Q: How do I undo the addition of a file?
```bash
git reset HEAD filename
```

### Q: How do I view commit history?
```bash
git log --oneline
```

### Q: How do I clone the project to another computer?
```bash
git clone https://github.com/YourUsername/albums-streamlit.git
cd albums-streamlit
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

---

## Recommended README.md Content

Your project already has a README.md, but you might want to add:

- Project screenshots
- Installation instructions
- Usage examples
- Contribution guidelines
- License information

---

## Next Steps

1. Run `./sync_to_github.sh` or manually execute the commands above.
2. View your repository on GitHub.
3. You can add README badges, GitHub Actions, etc.
4. Share it with others!

---

## Need Help?

If you encounter any issues, provide me with the error message and I'll help you resolve it!
