# Entrainement du classifieur sur Intel Arc XPU

Ce profil concerne le GPU Intel Arc expose par `torch.xpu`. Il ne cible pas le NPU
Intel, qui utilise une pile d'execution differente.

## 1. Environnement Windows

Utiliser un environnement separe des installations CUDA :

```powershell
python -m venv .venv-xpu
.\.venv-xpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements_XPU.txt
python -m pip install -e .
```

`requirements_3090Ti.txt` ne doit pas etre installe dans ce venv, car il impose les
wheels CUDA de PyTorch.

## 2. Verification

```powershell
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.get_device_name(0))"
referai-football inspect-hardware --hardware configs/hardware_XPU.yaml
```

La seconde commande doit afficher `backend=xpu` et `XPU:0`. Le profil explicite leve
une erreur si XPU n'est pas disponible au lieu de poursuivre silencieusement sur CPU.

## 3. Entrainement

```powershell
referai-football train-role --config configs/train_soccernet_roles.yaml --hardware configs/hardware_XPU.yaml
```

Le profil commence prudemment avec un batch de 4, aucun worker secondaire et AMP
desactivee. Ce reglage limite les allocations simultanees du backend Level Zero sous
Windows. Pour contourner les arrets natifs observes dans `torch_xpu.dll`, le pipeline
desactive aussi la memoire epinglee du DataLoader, le mode deterministe et les kernels
multi-tenseurs `foreach` de l'optimiseur AdamW. Ces changements ne concernent pas CUDA.
En cas de manque de memoire XPU, le code divise automatiquement le batch jusqu'a la
limite configuree. Une fois un premier entrainement stable, `amp: true` et un batch
plus grand peuvent etre testes dans une copie du profil.

Le meilleur checkpoint reste ecrit dans :

```text
runs/classify/soccernet_roles/weights/best.pt
```

Pour reprendre :

```powershell
referai-football train-role --config configs/train_soccernet_roles.yaml --hardware configs/hardware_XPU.yaml --resume
```
