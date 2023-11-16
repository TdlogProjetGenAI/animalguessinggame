// main_gen.js
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('generateForm').addEventListener('submit', function(event) {
        
        event.preventDefault();
        window.location.href = '/generate_image';
    });
});
