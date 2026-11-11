/**
 * Rising Waters: Flood Prediction System - Frontend Validation
 * Prevents form submission when any input field is empty.
 */

document.addEventListener("DOMContentLoaded", function () {
    // Get the prediction form element
    var form = document.getElementById("predictionForm");

    // Only attach validation if the form exists on the current page
    if (!form) {
        return;
    }

    // Listen for form submission
    form.addEventListener("submit", function (event) {
        // Collect all number input fields inside the form
        var inputs = form.querySelectorAll('input[type="number"]');
        var missingFields = [];

        // Check each input for empty or invalid values
        inputs.forEach(function (input) {
            var value = input.value.trim();
            if (value === "" || isNaN(parseFloat(value))) {
                missingFields.push(input.labels[0] ? input.labels[0].textContent : input.name);
                input.style.borderColor = "#c0392b";
            } else {
                input.style.borderColor = "";
            }
        });

        // If any field is missing, show alert and prevent submission
        if (missingFields.length > 0) {
            event.preventDefault();
            alert(
                "Please fill in all required fields.\n\nMissing or invalid fields:\n- " +
                    missingFields.join("\n- ")
            );
        }
    });
});
