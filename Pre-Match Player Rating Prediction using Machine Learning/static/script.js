const themes = {
    "Neymar": "/static/images/neymar.jpg",
    "N'golo Kante": "/static/images/kante.jpg",
    "Sergio Ramos": "/static/images/ramos.jpg"
};

function applyTheme() {
    const player = document.getElementById("playerSelect").value;
    const bg = document.getElementById("bgLayer");

    if (!player) {
        bg.style.backgroundImage = "url('/static/images/iconic.jpg')";
        return;
    }

    bg.style.backgroundImage = `url('${themes[player]}')`;
}

async function loadFixture() {
    const player = document.getElementById("playerSelect").value;
    if (!player) return;

    const response = await fetch("/fixture", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({player})
    });

    const data = await response.json();

    const box = document.getElementById("fixtureBox");
    box.innerHTML = `
        <strong>${data.date}</strong><br>
        vs ${data.opponent}<br>
        ${data.home ? "Home" : "Away"}
    `;

    box.classList.remove("hidden");
    document.getElementById("predictBtn").classList.remove("hidden");
}

async function predictRating() {
    const player = document.getElementById("playerSelect").value;

    const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({player})
    });

    const data = await response.json();

    console.log("SERVER RESPONSE:", data);   // 🔥 ADD THIS

    document.getElementById("resultBox").innerHTML =
        `Predicted Rating: ${data.predicted_rating}`;
}