import os
import shutil
import re

BASE_DIR = os.getcwd()

# Define Structure
STRUCTURE = {
    "assets/images": [],
    "assets/icons": [],
    "css": [],
    "js": [],
    "pages/auth": ["indexInicioDeSesion.html", "indexRegistrar.html", "indexRegistrarAutorizacion.html", "indexSeleccion.html", "POPup.html"],
    "pages/profile": ["Perfil.html", "PerfilEditar.html", "PerfilMenú.html", "PerfilEditarMenú.html", "PublicPerfil"],
    "pages/plans": ["PlanGratis.html", "PlanPremium.html", "PlanGratisMenú.html", "PlanPremiumMenú.html"],
    "pages/settings": [
        "Configuracion.html", "ConfiguracionMenú.html", "Configuracion_Despliege.html",
        "CuentaYSeguridad.html", "CuentaYSeguridad2.html",
        "Dispositivos Conectados.html", "Dispositivos Conectados Menú.html", "Dispositivos Conectados Agregado.html", "Dispositivos Conectados Agregar.html", "Dispositivos Conectados Mas.html", "Dispositivos Conectados MasMenú.html",
        "GestionDeCuenta.html", "GestionDeCuentaMenú.html", "GestionDeCuentaDespliegue.html", "GestionDeCuentaDespliegueMenú.html",
        "NotificacionesActivadas.html", "NotificacionesDesactivadas.html", "NotoficacionesDesactivadasMenú.html"
    ],
    "pages/info": [
        "SobreNosotros.html", "SobreNosotrosMenú.html", 
        "Contactanos.html", "ContactanosMenú.html", 
        "InformacionDeContacto.html", "InformacionDeContactoMenú.html"
    ],
    "pages/splash": [
        "index.html", 
        "PantallaDeCarga.html", "PantallaDeCarga2.html", "PantallaDeCarga3.html", 
        "pantalladecarga4.html", "pantalladecarga5.html", "PantallaDeCarga6.html"
    ],
    "pages/components": ["MenúHamburguesa.html", "Chat.html", "ChatOpciones.html", "ChatOpcionesMenú.html"]
}

# FILE MAPPING (Old Name -> New Rel Path)
FILE_MAP = {}
for category, files in STRUCTURE.items():
    for f in files:
        FILE_MAP[f] = f"{category}/{f}"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_files():
    # 1. CSS
    ensure_dir("css")
    for f in os.listdir("."):
        if f.endswith(".css"):
            shutil.move(f, f"css/{f}")
    
    # 2. JS (already in js/ folder, just verify)
    ensure_dir("js")
    # If there are stray js files
    for f in os.listdir("."):
        if f.endswith(".js"):
            shutil.move(f, f"js/{f}")
            
    # 3. Assets
    ensure_dir("assets/images")
    if os.path.exists("Assests"):
        for f in os.listdir("Assests"):
            shutil.move(f"Assests/{f}", f"assets/images/{f}")
        os.rmdir("Assests")
        
    if os.path.exists("public/images"):
        # Check for duplicates before moving
        for f in os.listdir("public/images"):
            src = f"public/images/{f}"
            dst = f"assets/images/{f}"
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree("public") # Remove public dir after moving

    # 4. Pages
    for category, files in STRUCTURE.items():
        if "pages/" in category:
            ensure_dir(category)
            for f in files:
                if os.path.exists(f):
                    shutil.move(f, f"{category}/{f}")

def update_paths():
    # Helper to calculate relative path
    # Target depth is always 2 for pages (pages/category/file.html) 
    # so ../../ gets to root
    
    for category, files in STRUCTURE.items():
        if "pages/" not in category: continue
        
        for filename in files:
            filepath = f"{category}/{filename}"
            if not os.path.exists(filepath): continue
            
            if os.path.isdir(filepath): continue # Skip folders like PublicPerfil

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Update CSS links
            content = re.sub(r'href="(?!http)([^"]+\.css)"', r'href="../../css/\1"', content)
            # Update JS links
            content = re.sub(r'src="(?!http)([^"]+\.js)"', r'src="../../js/\1"', content)
            content = re.sub(r'src="js/([^"]+)"', r'src="../../js/\1"', content) # Handle js/ prefix
            
            # Update Images
            # Old: public/images/..., Assests/..., img/...
            content = content.replace('src="public/images/', 'src="../../assets/images/')
            content = content.replace('src="Assests/', 'src="../../assets/images/')
            content = content.replace('src="img/', 'src="../../assets/images/')
            
            # Update Links to other HTML files
            def replace_link(match):
                target = match.group(1)
                if target in FILE_MAP:
                    # Generic way: from pages/cat1 to pages/cat2
                    # ../cat2/target
                    new_target = FILE_MAP[target]
                    # Since we are in pages/cat1, we need ../ + cat2/file
                    cat_level = category.split("/")[-1] 
                    return f'href="../{new_target.replace("pages/", "")}"'
                return match.group(0)

            content = re.sub(r'href="([^"]+\.html)"', replace_link, content)
            # Handle onclick location.href
            content = re.sub(r"location.href\s*=\s*['\"]([^'\"]+\.html)['\"]", lambda m: f"location.href='../{FILE_MAP.get(m.group(1), m.group(1)).replace('pages/', '')}'", content)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

if __name__ == "__main__":
    move_files()
    update_paths()
    print("Reorganization Complete")
