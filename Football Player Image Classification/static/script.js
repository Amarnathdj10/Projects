function uploadImage() {

    let fileInput = document.getElementById("imageInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select an image first!");
        return;
    }

    // Show preview
    let reader = new FileReader();
    reader.onload = function(e) {
        let preview = document.getElementById("preview");
        preview.src = e.target.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(file);

    let formData = new FormData();
    formData.append("image", file);

    document.getElementById("result").innerText = "Analyzing...";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerText = data.error;
        } else {
            document.getElementById("result").innerText =
                "Prediction: " + data.prediction;
        }
    })
    .catch(() => {
        document.getElementById("result").innerText =
            "Something went wrong.";
    });
}