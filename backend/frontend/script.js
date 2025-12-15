let villageData = {};

async function loadVillages() {
  const res = await fetch("village_coords.json");
  villageData = await res.json();

  const villageSelect = document.getElementById("village");
  Object.keys(villageData).forEach(village => {
    const opt = document.createElement("option");
    opt.value = village;
    opt.textContent = village;
    villageSelect.appendChild(opt);
  });
}

document.getElementById("village").addEventListener("change", (e) => {
  const village = e.target.value;
  if (village && villageData[village]) {
    const [lat, lon] = villageData[village];
    document.getElementById("lat").value = lat;
    document.getElementById("lon").value = lon;
  }
});

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
    const res = await fetch("http://127.0.0.1:5000/predict", {
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
    output.innerHTML = "⚠️ Cannot connect to backend. Make sure Flask is running.";
  }
});

loadVillages();
