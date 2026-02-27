function uploadImage() {

    let fileInput = document.getElementById("imageInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select an image first!");
        return;
    }

    let formData = new FormData();
    formData.append("image", file);

    document.getElementById("result").innerText = "Predicting...";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText =
            "Prediction: " + data.prediction;
    })
    .catch(error => {
        document.getElementById("result").innerText =
            "Error occurred. Try again.";
    });
}