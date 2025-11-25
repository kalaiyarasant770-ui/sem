// login.js

document.addEventListener("DOMContentLoaded", () => {

  // Check if already logged in
  if (localStorage.getItem("loggedIn") === "true") {
    console.log("User already logged in — redirecting to index.html");
    window.location.href = "index.html";
    return;
  }

  const form = document.getElementById("loginForm");
  const message = document.getElementById("message");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value.trim();

    console.log("Login attempt:", username);

    // ✅ Call backend /login API
    try {
      const response = await fetch(`${ACTIVE_CONFIG.BASE_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      console.log(response)

      // const data = await response.json();

      let data;
      try {
        data = await response.json();
      } catch {
        const text = await response.text();
        console.error("⚠️ Non-JSON response:", text);
        throw new Error("Server returned invalid response");
      }

      if (response.ok && data.success) {
        // ✅ Store session data
        localStorage.setItem("loggedIn", "true");
        localStorage.setItem("username", data.username);
        localStorage.setItem("vertical_id", data.vertical_id);
        localStorage.setItem("vertical_name", data.vertical_name);

        message.style.display = "block";
        message.textContent = `Welcome ${data.vertical_name}! Redirecting...`;
        message.style.color = "green";

        setTimeout(() => {
          window.location.href = "index.html";
        }, 500);
      } else {
        message.style.display = "block";
        message.textContent = data.message || "Invalid credentials!";
        message.style.color = "red";
      }
    } catch (error) {
      console.error("Login error:", error);
      message.style.display = "block";
      message.textContent = "Login failed. Please try again.";
      message.style.color = "red";
    }
  });
});