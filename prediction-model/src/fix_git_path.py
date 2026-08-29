import os
import sys
import subprocess
import winreg

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

git_cmd = r"C:\Users\Kloudtech Software\AppData\Local\MinGit\cmd"
git_bin = r"C:\Users\Kloudtech Software\AppData\Local\MinGit\mingw64\bin"

# 1. Update Windows Registry User PATH
try:
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS)
    current_path, _ = winreg.QueryValueEx(key, "Path")
    if git_cmd not in current_path:
        new_path = f"{git_cmd};{git_bin};{current_path}"
        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
        print("✓ Successfully added MinGit to User PATH in Windows Registry!")
    else:
        print("✓ MinGit already present in User PATH.")
    winreg.CloseKey(key)
except Exception as e:
    print(f"Error updating registry PATH: {e}")

# 2. Update current process PATH
os.environ["PATH"] = f"{git_cmd};{git_bin};" + os.environ.get("PATH", "")

# 3. Test git and git-credential-manager
res_git = subprocess.run(["git", "--version"], capture_output=True, text=True)
print(f"Git version: {res_git.stdout.strip()}")

res_gcm = subprocess.run(["git-credential-manager", "--version"], capture_output=True, text=True)
print(f"GCM version: {res_gcm.stdout.strip()}")
