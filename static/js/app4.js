document.addEventListener('DOMContentLoaded', function() {
    // Check if there's a flashed message with class 'error'
    var errorMessage = document.querySelector('.flashed-message.error');
    if (errorMessage) {
        // Display an alert with the error message
        alert(errorMessage.textContent);
    }
});

        
    