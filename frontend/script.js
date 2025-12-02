document.getElementById("analyzeBtn").addEventListener("click", async () => {
    const fileInput = document.getElementById("resumeFile");
    const results = document.getElementById("results");

    if (!fileInput.files.length) {
        results.innerHTML = "<p>Please upload a file first.</p>";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    results.innerHTML = "<p>Analyzing...</p>";

    try {
        const res = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        results.innerHTML = `
            <div class="analysis-container">

                <h3>Applicant Information</h3>
                <p>Name: ${data.name || "Not found"}</p>
                <p>Email: ${data.email || "Not found"}</p>

                <h3>Career Field Prediction</h3>
                <p>${data.field} (${(data.confidence * 100).toFixed(1)}%)</p>

                <h3>Skill Recommendations</h3>
                <ul class="improvements-list">
                    ${data.dqn_skill.map(s => `<li>${s}</li>`).join("")}
                </ul>

                <h3>Resume Quality Score</h3>
                <p>${data.score} / 100</p>

                <h3>Actionable Improvements</h3>
                <ul class="improvements-list">
                    ${data.mdp_actions.map(a =>
                        `<li>${a.section}: ${a.action}</li>`
                    ).join("")}
                </ul>

            </div>
        `;
    } catch (err) {
        console.error(err);
        results.innerHTML = "<p>Error analyzing resume.</p>";
    }
});
