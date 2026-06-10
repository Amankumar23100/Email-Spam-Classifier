async function checkSpam() {

    let msg = document.getElementById("message").value.trim();
    let result = document.getElementById("result");

    if (msg === "") {
        result.innerText = "Please enter email text.";
        result.style.color = "orange";
        return;
    }

    try {
        let res = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: msg })
        });

        let data = await res.json();

        result.innerText = "Prediction: " + data.prediction;

        if (data.prediction.includes("Spam")) {
            result.style.color = "red";
        } else {
            result.style.color = "green";
        }

    } catch (error) {
        result.innerText = "Server Error";
        result.style.color = "red";
    }
}