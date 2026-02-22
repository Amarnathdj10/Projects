const themes = {
    messi: {
        gradient: "linear-gradient(135deg, #004d98, #a50044)",
        image: "client/images/messi.png"
    },
    neymar: {
        gradient: "linear-gradient(135deg, #002654, #ffcc00)",
        image: "client/images/neymar.png"
    },
    kante: {
        gradient: "linear-gradient(135deg, #034694, #ffffff)",
        image: "client/images/kante.png"
    },
    ramos: {
        gradient: "linear-gradient(135deg, #8b0000, #ffffff)",
        image: "client/images/ramos.png"
    }
};

function changeTheme() {

    const player = document.getElementById("playerSelect").value;
    const body = document.body;
    const overlay = document.querySelector(".overlay");

    if (!player || !themes[player]) {
        body.style.background = "linear-gradient(135deg, #111, #222)";
        overlay.style.backgroundImage = "none";
        return;
    }

    body.style.background = themes[player].gradient;

    overlay.style.backgroundImage = `url(${themes[player].image})`;
    overlay.style.opacity = "0.25";
}

function predictFixture() {

    const player = document.getElementById("playerSelect").value;
    const resultBox = document.getElementById("predictionResult");

    if (!player) {
        alert("Please select a player first.");
        return;
    }

    resultBox.innerHTML = "Predicting...";
    resultBox.style.opacity = "0.6";

    setTimeout(() => {

        const predictedRating = (Math.random() * 1.5 + 7.2).toFixed(2);

        resultBox.style.opacity = "1";
        resultBox.innerHTML = `
            <div class="prediction-card">
                <h3>Upcoming Fixture Prediction</h3>
                <div class="rating-value">${predictedRating}</div>
            </div>
        `;

    }, 1000);
}