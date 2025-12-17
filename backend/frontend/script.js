document.getElementById("predict").addEventListener("click", async () => {
  const lat = parseFloat(document.getElementById("lat").value);
  const lon = parseFloat(document.getElementById("lon").value);
  const output = document.getElementById("output");

  if (isNaN(lat) || isNaN(lon)) {
    alert("⚠️ Please select a valid district!");
    return;
  }

  output.innerHTML = "⏳ Fetching satellite data and predicting...";

  try {
    const res = await fetch("https://newproject-10-80xp.onrender.com/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ lat, lon })
    });

    if (!res.ok) throw new Error(`Server responded with ${res.status}`);
    const data = await res.json();

    if (data.error) {
      output.innerHTML = `❌ Error: ${data.error}`;
    } else {
      output.innerHTML = `
        <h3>🌿 Predicted Nutrient Levels</h3>
        <p><strong>Nitrogen (N):</strong> ${data.Nitrogen_avg.toFixed(2)} kg/ha</p>
        <p><strong>Phosphorus (P):</strong> ${data.Phosphorus_avg.toFixed(2)} kg/ha</p>
        <p><strong>Potassium (K):</strong> ${data.Potassium_avg.toFixed(2)} kg/ha</p>
        <hr>
        <h3>💡 Fertilizer Suggestion</h3>
        <p>🧪 ${data.Suggestions.Nitrogen}</p>
        <p>🌾 ${data.Suggestions.Phosphorus}</p>
        <p>🌻 ${data.Suggestions.Potassium}</p>
      `;
    }
  } catch (err) {
    console.error("Backend error:", err);
    output.innerHTML = "⚠️ Cannot connect to backend. Make sure Render backend is live.";
  }
});
