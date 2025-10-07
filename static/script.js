document.addEventListener("DOMContentLoaded", () => {
  const checkBtn = document.getElementById("checkBtn");
  const clearBtn = document.getElementById("clearBtn");
  const userText = document.getElementById("userText");
  const resultCard = document.getElementById("result");
  const scoreText = document.getElementById("scoreText");
  const meterFill = document.getElementById("meterFill");
  const breakdown = document.getElementById("breakdown");

  checkBtn.addEventListener("click", async () => {
    const text = userText.value.trim();
    if (!text) {
      alert("Please enter text");
      return;
    }

    // Reset UI for new request
    scoreText.textContent = "Analyzing...";
    resultCard.style.display = "block";
    meterFill.style.width = "0%";
    breakdown.textContent = "";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const result = await response.json();
      if (result.error) {
        alert(result.error);
        resultCard.style.display = "none";
        return;
      }

      // ------------------------------
      // Display classification scores
      // ------------------------------
      scoreText.textContent = `Human: ${result.human_score}% | AI: ${result.ai_score}% (${result.label})`;
      meterFill.style.width = result.ai_score + "%";

      // ------------------------------
      // Sentiment display + emoji + color
      // ------------------------------
      let sentimentEmoji = "😐";
      let sentimentColor = "gray";
      if (result.sentiment === "Positive") {
        sentimentEmoji = "😊";
        sentimentColor = "green";
      } else if (result.sentiment === "Negative") {
        sentimentEmoji = "😡";
        sentimentColor = "red";
      }

      // ------------------------------
      // Display metric breakdown
      // ------------------------------
      breakdown.innerHTML = `
        <li><strong>Readability:</strong> ${result.readability}</li>
        <li><strong>Vocabulary Diversity:</strong> ${result.vocab_diversity}</li>
        <li><strong>Average Sentence Length:</strong> ${result.avg_sentence_length} words</li>
        <li><strong>Sentiment:</strong> <span id="sentimentVal" style="color:${sentimentColor};">${result.sentiment} ${sentimentEmoji}</span></li>
      `;
    } catch (err) {
      console.error(err);
      alert("Error connecting to server.");
      resultCard.style.display = "none";
    }
  });

  // Clear text and result
  clearBtn.addEventListener("click", () => {
    userText.value = "";
    resultCard.style.display = "none";
  });
});
