# GitHub Repository Setup Guide

Step-by-step guide to publish this repository on GitHub.

## Prerequisites

- GitHub account
- Git installed locally
- Repository code ready (this directory)

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in repository details:
   - **Name**: `head-trajectories` (or your preferred name)
   - **Description**: "Developmental Trajectories of Attention Heads - Mechanistic Interpretability Research"
   - **Visibility**: Public (recommended for research)
   - **Initialize**: Do NOT initialize with README, .gitignore, or license (we have these)
3. Click "Create repository"

## Step 2: Update Repository URLs

✅ **Already done!** All URLs have been updated to:
- GitHub: `https://github.com/abderahmane-ai/head-trajectories`
- Username: `abderahmane-ai`

Files updated:
- ✅ `README.md` - Badge URLs and clone command
- ✅ `setup.py` - Package URL
- ✅ `docs/QUICKSTART.md` - Clone command
- ✅ All documentation files

## Step 3: Initialize Git Repository

```bash
cd /path/to/head-trajectories

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Developmental Trajectories v1.0.0

- Complete transformer implementation
- Five behavioral scoring functions
- Probing pipeline with resumption
- Comprehensive test suite (60 tests)
- Full documentation
- CI/CD with GitHub Actions"
```

## Step 4: Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/abderahmane-ai/head-trajectories.git

# Push to main branch
git branch -M main
git push -u origin main
```

## Step 5: Configure Repository Settings

### General Settings
1. Go to repository Settings
2. Under "General":
   - Add website URL (if you have one)
   - Add topics: `mechanistic-interpretability`, `transformers`, `attention-heads`, `pytorch`, `research`
   - Enable "Issues"
   - Enable "Discussions" (optional but recommended)

### Branch Protection (Optional but Recommended)
1. Go to Settings → Branches
2. Add rule for `main` branch:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass (Tests workflow)
   - ✅ Require branches to be up to date

### GitHub Actions
1. Go to Actions tab
2. Enable workflows
3. The test workflow should run automatically on push

### Pages (Optional - for documentation)
1. Go to Settings → Pages
2. Source: Deploy from branch
3. Branch: `main`, folder: `/docs`
4. Your docs will be at: `https://YOUR_USERNAME.github.io/head-trajectories/`

## Step 6: Add Repository Metadata

### About Section
Click the gear icon next to "About" and add:
- **Description**: "Developmental Trajectories of Attention Heads - Mechanistic Interpretability Research"
- **Website**: Your website or paper link (if available)
- **Topics**: 
  - `mechanistic-interpretability`
  - `transformers`
  - `attention-mechanisms`
  - `pytorch`
  - `deep-learning`
  - `research`
  - `interpretability`
  - `language-models`

### Social Preview
1. Go to Settings → General
2. Scroll to "Social preview"
3. Upload an image (1280×640px recommended)
   - Could be a figure from your results
   - Or create a banner with the project title

## Step 7: Create Initial Release

1. Go to Releases → "Create a new release"
2. Tag: `v1.0.0`
3. Title: "Initial Release - v1.0.0"
4. Description:
```markdown
## Developmental Trajectories v1.0.0

First public release of the research codebase.

### Features
- ✅ Complete LLaMA-style transformer implementation
- ✅ Five behavioral scoring functions
- ✅ Head classification with tie-breaking
- ✅ Probing pipeline with resumption
- ✅ Comprehensive test suite (60 tests, 100% passing)
- ✅ Full documentation and guides
- ✅ CI/CD with GitHub Actions

### Performance
- 30-100× speedup on semantic scoring (vectorized)
- Incremental checkpoint saving
- Safe torch.load operations

### Documentation
- Quick start guide
- Architecture overview
- API documentation
- FAQ
- Contributing guidelines

### Installation
```bash
git clone https://github.com/abderahmane-ai/head-trajectories.git
cd head-trajectories
pip install -r requirements.txt
python run_tests.py
```

See [QUICKSTART.md](docs/QUICKSTART.md) for detailed setup instructions.
```

5. Click "Publish release"

## Step 8: Set Up Discussions (Optional)

1. Go to Settings → General
2. Enable "Discussions"
3. Create categories:
   - **Q&A**: Questions about usage
   - **Ideas**: Feature requests and suggestions
   - **Show and tell**: Share your results
   - **General**: General discussion

## Step 9: Add Collaborators (Optional)

If you have collaborators:
1. Go to Settings → Collaborators
2. Add collaborators by username
3. Set appropriate permissions

## Step 10: Promote Your Repository

### README Badges
The README already includes badges. Update them:
- Tests badge: Will work automatically once Actions run
- Python version: Already correct
- License: Already correct

### Share
- Tweet about it (tag @anthropicai, @neelnanda_io if relevant)
- Post on relevant subreddits (r/MachineLearning, r/LanguageTechnology)
- Share in ML Discord servers
- Add to your personal website/portfolio
- Submit to Papers with Code (when paper is published)

### Academic
- Add to your CV/resume
- Include in your thesis/dissertation
- Reference in future papers
- Present at conferences/seminars

## Verification Checklist

Before announcing publicly, verify:

- [ ] All tests pass on GitHub Actions
- [ ] README displays correctly
- [ ] Links work (clone URL, badges, etc.)
- [ ] Documentation is accessible
- [ ] License is correct
- [ ] No sensitive information in commit history
- [ ] Repository description and topics are set
- [ ] Issues and Discussions are enabled
- [ ] Branch protection is configured (if desired)
- [ ] Initial release is created

## Maintenance

### Regular Tasks
- Respond to issues within 48 hours
- Review PRs within 1 week
- Update dependencies quarterly
- Keep documentation current

### Monitoring
- Watch for security alerts
- Monitor GitHub Actions for failures
- Check for outdated dependencies
- Review and merge Dependabot PRs

## Getting Help

If you encounter issues:
1. Check GitHub's documentation: https://docs.github.com
2. Search GitHub Community: https://github.community
3. Ask in relevant Discord servers
4. Open an issue in a similar project for advice

## Next Steps

After publishing:
1. Monitor initial feedback
2. Respond to first issues/questions
3. Consider writing a blog post
4. Prepare paper submission (if applicable)
5. Engage with the community

## Example Announcement

### Twitter/X
```
🚀 Just released: Developmental Trajectories of Attention Heads

A complete research codebase for studying when & how attention heads specialize during training.

✅ 60 tests, 100% passing
✅ Full documentation
✅ Reproducible experiments
✅ MIT licensed

https://github.com/abderahmane-ai/head-trajectories

#MachineLearning #Interpretability #PyTorch
```

### Reddit (r/MachineLearning)
```
[R] Developmental Trajectories of Attention Heads - Open Source Research Code

I'm sharing my research codebase for studying attention head development in transformers. The project tracks 64 heads across 100 checkpoints to understand their developmental trajectories.

Key features:
- Complete transformer implementation from scratch
- Five behavioral scoring functions
- Comprehensive test suite (60 tests)
- Full documentation and guides
- Reproducible experiments (~$25 on Modal)

GitHub: https://github.com/abderahmane-ai/head-trajectories

Feedback and contributions welcome!
```

Good luck with your repository! 🎉
