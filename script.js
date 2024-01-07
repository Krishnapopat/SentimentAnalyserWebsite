function analyzeSentiment() {
    var sentence = document.getElementById("sentence").value;

    // Send the sentence to the server for analysis.
    fetch('/analyze_sentiment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'sentence=' + encodeURIComponent(sentence),
    })
    .then(response => response.json())
    .then(data => displayResult(data.sentiment))
    .catch(error => console.error('Error:', error));
}

function displayResult(sentiment) {
    var resultDiv = document.getElementById("result");
    var sentimentElement = document.getElementById("sentiment");

    resultDiv.classList.remove("hidden");
    sentimentElement.textContent = "The Sentiment of the Entered Sentence is : " + sentiment ;
}
