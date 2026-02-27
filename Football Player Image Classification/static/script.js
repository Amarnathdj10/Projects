document.addEventListener("DOMContentLoaded", function () {

    let selectedFile = null;

    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("imageInput");

    if (!dropZone || !fileInput) {
        console.error("Drop zone elements not found.");
        return;
    }

    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        selectedFile = e.target.files[0];
        showPreview(selectedFile);
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        selectedFile = e.dataTransfer.files[0];
        showPreview(selectedFile);
    });

    function showPreview(file) {
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            const preview = document.getElementById("preview");
            preview.src = e.target.result;
            preview.style.display = "block";
        };
        reader.readAsDataURL(file);
    }

    window.uploadImage = function () {

        if (!selectedFile) {
            alert("Please select an image first!");
            return;
        }

        let formData = new FormData();
        formData.append("image", selectedFile);

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
    };

});