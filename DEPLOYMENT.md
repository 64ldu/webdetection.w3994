# 🚀 Deployment Guide

## Option 1: GitHub Pages (Free Static Hosting)

### Step 1: Create GitHub Repository
1. Go to [GitHub](https://github.com) and sign in
2. Click **"New repository"** (green button on the right)
3. Repository name: `face-ai-web` (or your preferred name)
4. Description: `Real-time face detection and expression recognition web app`
5. Choose **Public** (required for GitHub Pages)
6. **Don't** initialize with README (we already have one)
7. Click **"Create repository"**

### Step 2: Push to GitHub
```bash
# Add your GitHub repository as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/face-ai-web.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy to GitHub Pages
1. Go to your repository on GitHub
2. Click **Settings** tab
3. Scroll down to **Pages** section
4. Under "Build and deployment", select **GitHub Actions**
5. Create a new file `.github/workflows/deploy.yml`

### Step 4: Create GitHub Actions Workflow
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./
```

---

## Option 2: Heroku (Free Web App Hosting)

### Step 1: Install Heroku CLI
```bash
# On macOS with Homebrew
brew install heroku/brew/heroku

# Or download from https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Login to Heroku
```bash
heroku login
```

### Step 3: Create Heroku App
```bash
heroku create your-app-name
```

### Step 4: Create Procfile
Create a file named `Procfile` (no extension):
```
web: python app.py
```

### Step 5: Deploy
```bash
git add .
git commit -m "Add Heroku deployment"
git push heroku main
```

---

## Option 3: PythonAnywhere (Free Python Hosting)

### Step 1: Sign Up
1. Go to [PythonAnywhere](https://www.pythonanywhere.com)
2. Create a free account

### Step 2: Create Web App
1. Go to **Web** tab
2. Click **"Add a new web app"**
3. Choose **Flask** and **Python 3.9**
4. Set path to: `/home/yourusername/mysite`

### Step 3: Upload Code
1. Go to **Files** tab
2. Upload your project files
3. Or use git to clone your repository

### Step 4: Configure
1. Edit **WSGI configuration file**
2. Update path to your app.py
3. Reload web app

---

## Option 4: Vercel (Modern Web Hosting)

### Step 1: Install Vercel CLI
```bash
npm i -g vercel
```

### Step 2: Deploy
```bash
vercel
```

### Step 3: Configure for Python
Create `vercel.json`:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

---

## 🎯 Recommended: GitHub Pages + Replit

### Easiest Option:
1. **Push to GitHub** (steps above)
2. **Go to Replit.com** → Import from GitHub
3. **Run** → Share link

### Why this combo:
- ✅ **Free**
- ✅ **No setup required**
- ✅ **Instant deployment**
- ✅ **Supports WebSockets**
- ✅ **Custom domain support**

---

## 🔗 After Deployment

### Test Your App
1. Visit your deployed URL
2. Click **"Start Camera"**
3. Grant camera permissions
4. Test face detection

### Share Your App
- Share the URL with friends
- Add to your portfolio
- Submit to tech showcases

### Custom Domain (Optional)
- GitHub Pages: Configure in Settings → Pages
- Heroku: `heroku domains:add yourdomain.com`
- Vercel: Dashboard → Domains

---

## 🛠️ Troubleshooting

### Camera Not Working?
- Check HTTPS (required for camera access)
- Verify browser permissions
- Test in different browsers

### Deployment Issues?
- Check logs in deployment platform
- Verify requirements.txt
- Check port configuration

### Performance Issues?
- Add rate limiting
- Optimize image processing
- Consider CDN for static assets

---

## 📱 Mobile Support

The web app works on mobile devices:
- Responsive design
- Touch-friendly interface
- Camera access on iOS/Android

---

## 🔒 Security Notes

- Camera access requires HTTPS
- Add rate limiting for production
- Consider user authentication
- Monitor for abuse

---

Choose the option that best fits your needs. For beginners, I recommend **GitHub + Replit** for the easiest deployment!
