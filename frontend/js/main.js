async function registrarUsuario(username, email, password, plan = "Gratis", termsAccepted = null) {
  try {
      const emailNorm = email.trim().toLowerCase();
         const fullName = username.trim();
      const storedAccepted =
        localStorage.getItem("termsAccepted") ||
        localStorage.getItem("privacyAccepted") ||
        localStorage.getItem("termsRead");
      const finalTermsAccepted = termsAccepted !== null
        ? !!termsAccepted
        : (storedAccepted === "true");
      const termsVersion = localStorage.getItem("termsVersion") || "2025-04-07";
      if (!finalTermsAccepted) {
        showToast("Debes aceptar los Términos y Condiciones", false);
        return;
      }
      
    const res = await fetch(API_URL + "register/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: emailNorm,     // username técnico = email
          email: emailNorm,
          full_name: fullName,     // nombre real
          password,
          plan,
          termsAccepted: finalTermsAccepted,
          termsVersion,
          termsSource: "web",
        }),

    });

    const data = await res.json();

    if (res.ok) {
        showToast("✅ Usuario registrado: " + data.user.username);


      console.log("✅ Respuesta del backend:", data);
      setTimeout(() => {
        window.location.href = "indexInicioDeSesion.html";
      }, 1500);
    } else {
      console.error("❌ Error del backend:", data);
      showToast("❌ " + (data.error || "No se pudo registrar el usuario"), false);
    }
  } catch (err) {
    console.error("⚠ Error de conexión:", err);
    showToast("⚠ No se pudo conectar con el servidor", false);
  }
}
function cerrarSesion() {
    console.log("✅ cerrarSesion() ejecutada");
  // 1. Limpiar datos de sesión
    localStorage.removeItem("access");
    localStorage.removeItem("refresh");
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    localStorage.removeItem("username");
    localStorage.removeItem("isLoggedIn");

  // 2. Cerrar el menú si existe
  if (typeof toggleMenu === "function") {
    toggleMenu();
  }

  // 3. Redirigir al login
  window.location.href = "../pages/auth/indexInicioDeSesion.html";
}
