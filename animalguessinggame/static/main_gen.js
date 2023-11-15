// main_gen.js
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('generateForm').addEventListener('submit', function(event) {
        event.preventDefault();  // Empêche l'envoi du formulaire par défaut

        // Redirige vers la page de génération d'image (changez l'URL si nécessaire)
        window.location.href = '/generate_image';
    });
});
