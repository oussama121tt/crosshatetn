# Publie le dépôt sur GitHub (privé) et déploie la démo web sur Vercel, en une commande.
#   Double-clic sur deploy.bat, ou :  powershell -ExecutionPolicy Bypass -File deploy.ps1
# La première fois, deux connexions s'ouvrent dans le navigateur : GitHub puis Vercel. Ensuite, tout est automatique.
param(
    [string]$Compte = "oussama121tt",
    [string]$Depot = "crosshatetn"
)
$ErrorActionPreference = "Continue"   # les outils écrivent des avertissements sur stderr : on teste les codes de sortie
function Ok($quoi) { if ($LASTEXITCODE -ne 0) { throw "Échec : $quoi (code $LASTEXITCODE)" } }
function Teste($ligne) { cmd /c "$ligne >nul 2>&1"; return ($LASTEXITCODE -eq 0) }
Set-Location -Path $PSScriptRoot

function Etape($texte) { Write-Host "`n==> $texte" -ForegroundColor Cyan }
function Rafraichir-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
function Installer($commande, $idWinget) {
    if (-not (Get-Command $commande -ErrorAction SilentlyContinue)) {
        Write-Host "$commande absent : installation avec winget ($idWinget)..."
        winget install --id $idWinget -e --silent --accept-package-agreements --accept-source-agreements
        Rafraichir-Path
        if (-not (Get-Command $commande -ErrorAction SilentlyContinue)) {
            throw "$commande installé mais introuvable : fermez cette fenêtre, rouvrez-la et relancez deploy.bat."
        }
    }
}

Etape "Vérification des outils (git, GitHub CLI, Node.js)"
Installer "git" "Git.Git"
Installer "gh" "GitHub.cli"
Installer "node" "OpenJS.NodeJS.LTS"

Etape "Contrôle : aucun corpus ni secret ne doit partir sur GitHub"
if (-not (Test-Path "web\model\modele_demo.npz")) { throw "web\model\modele_demo.npz manquant : lancer d'abord  python -m crosshatetn export-web" }

Etape "Connexion à GitHub"
if (-not (Teste "gh auth status")) { gh auth login --hostname github.com --git-protocol https --web; Ok "connexion GitHub" }
gh auth setup-git; Ok "configuration de git avec GitHub"

Etape "Commit local"
if (-not (Test-Path ".git")) { git init | Out-Null; Ok "git init" }
git config user.name  $Compte
if (-not (git config user.email)) { git config user.email "$Compte@users.noreply.github.com" }
git add -A
$suivis = git diff --cached --name-only
foreach ($f in $suivis) {
    if ($f -match '^(data/|outputs/)' -or $f -match '(^|/)\.env$') { throw "Fichier interdit dans le commit : $f" }
}
if (-not (Teste "git diff --cached --quiet")) { git commit -m "CrossHateTN Bronze : socle, baselines, démo locale et démo web" | Out-Null; Ok "commit" }
git branch -M main

Etape "Dépôt GitHub privé $Compte/$Depot"
if (-not (Teste "gh repo view $Compte/$Depot")) {
    gh repo create "$Compte/$Depot" --private --source . --remote origin --push; Ok "création du dépôt GitHub"
} else {
    if (-not (git remote | Select-String -Quiet "^origin$")) { git remote add origin "https://github.com/$Compte/$Depot.git" }
    git push -u origin main; Ok "git push"
}
$visibilite = gh repo view "$Compte/$Depot" --json visibility -q .visibility
Write-Host "Dépôt : https://github.com/$Compte/$Depot ($visibilite)"

Etape "Déploiement de la démo web sur Vercel (dossier web)"
Push-Location web
try {
    if (-not (Teste "npx --yes vercel@latest whoami")) { npx --yes vercel@latest login; Ok "connexion Vercel" }
    if (-not (Test-Path ".vercel\project.json")) { npx --yes vercel@latest link --yes --project $Depot; Ok "création du projet Vercel" }
    $sortie = npx --yes vercel@latest deploy --prod --yes 2>&1
    $code = $LASTEXITCODE
    $sortie | ForEach-Object { Write-Host $_ }
    if ($code -ne 0) { throw "Échec du déploiement Vercel (voir ci-dessus)" }
    $url = ($sortie | Select-String -Pattern 'https://[^\s]+\.vercel\.app' -AllMatches).Matches | Select-Object -Last 1
} finally { Pop-Location }

Etape "Terminé"
if ($url) {
    Write-Host "Démo en ligne : $($url.Value)" -ForegroundColor Green
    Start-Process $url.Value
} else {
    Write-Host "Déploiement lancé : l'adresse est dans les lignes ci-dessus (https://...vercel.app)." -ForegroundColor Yellow
}
Write-Host "Pour redéployer après une modification : relancer deploy.bat."
