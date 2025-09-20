import streamlit as st
import joblib
import numpy as np
import streamlit.components.v1 as components

# --- PATHS AND CONFIGURATION ---
MODEL_PATH = "models/crop_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# --- HTML TEMPLATES ---
WELCOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to Crop Recommender</title>
    <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
    <style type="text/tailwindcss">
        :root {
            --primary-500: #22c55e;
            --primary-600: #16a34a;
            --primary-700: #15803d;
        }
        body { font-family: 'Work Sans', 'Noto Sans', sans-serif; }
    </style>
</head>
<body class="bg-gray-50 flex items-center justify-center min-h-screen">
    <div class="max-w-md w-full p-8 bg-white rounded-lg shadow-xl text-center">
        <h1 class="text-4xl font-bold text-green-700 mb-4">Welcome to Agri-GPT</h1>
        <p class="text-lg text-gray-600 mb-6">Your smart companion for modern farming.</p>
        <p class="text-gray-500 mb-8">This AI-powered system recommends the most suitable crop for your land based on soil and weather conditions.</p>
        <button onclick="window.location.href='?continue=yes'" class="w-full px-6 py-3 rounded-full bg-green-600 text-white text-lg font-semibold hover:bg-green-700 transition-colors">
            Continue
        </button>
    </div>
</body>
</html>
"""

PROFILE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <link crossorigin="" href="https://fonts.gstatic.com/" rel="preconnect"/>
    <link as="style" href="https://fonts.googleapis.com/css2?display=swap&family=Noto+Sans:wght@400;500;700;900&family=Work+Sans:wght@400;500;700;900" onload="this.rel='stylesheet'" rel="stylesheet"/>
    <title>Farm Profile Setup</title>
    <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
    <style type="text/tailwindcss">
      :root {
        --primary-500: #22c55e;
        --primary-600: #16a34a;
        --primary-700: #15803d;
      }
      body { min-height: max(884px, 100dvh); }
    </style>
</head>
<body class="bg-gray-50">
<div class="relative flex h-auto min-h-screen w-full flex-col bg-white" style='font-family: "Work Sans", "Noto Sans", sans-serif;'>
  <header class="sticky top-0 z-10 bg-white border-b border-gray-200">
    <div class="mx-auto flex h-16 max-w-md items-center justify-between px-4">
      <button onclick="window.location.href='?back=yes'" class="text-gray-700 hover:text-gray-900">
        ◀
      </button>
      <h1 class="text-lg font-bold text-gray-900">Farm Profile Setup</h1>
      <div class="w-6"></div>
    </div>
  </header>

  <main class="flex-1 overflow-y-auto">
    <div class="mx-auto max-w-md p-4">
      <div class="mb-6">
        <div class="flex justify-between mb-2">
          <p class="text-sm font-medium text-green-600">Step 1 of 2</p>
          <p class="text-sm font-medium text-gray-500">Personal Details</p>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-2">
          <div class="bg-green-600 h-2 rounded-full" style="width: 50%;"></div>
        </div>
      </div>

      <div class="space-y-6">
        <div class="space-y-4">
          <h2 class="text-base font-semibold text-gray-800">Your Information</h2>
          <div class="space-y-2">
            <label class="text-sm font-medium text-gray-700" for="farmer-name">Farmer's Name</label>
            <input class="form-input block w-full rounded-md border-gray-300 shadow-sm" id="farmer-name" placeholder="Enter your full name" type="text"/>
          </div>
          <div class="space-y-2">
            <label class="text-sm font-medium text-gray-700" for="email">Email Address</label>
            <input class="form-input block w-full rounded-md border-gray-300 shadow-sm" id="email" placeholder="you@example.com" type="email"/>
          </div>
          <div class="space-y-2">
            <label class="text-sm font-medium text-gray-700" for="contact-number">Contact Number</label>
            <input class="form-input block w-full rounded-md border-gray-300 shadow-sm" id="contact-number" placeholder="Enter your phone number" type="tel"/>
          </div>
        </div>
      </div>
    </div>
  </main>

  <footer class="sticky bottom-0 bg-white border-t border-gray-200">
    <div class="mx-auto max-w-md p-4 flex items-center space-x-4">
      <button onclick="window.location.href='?back=yes'"
          class="w-full rounded-md border border-gray-300 bg-white px-4 py-3 text-base font-semibold text-gray-700">
        Back
      </button>
      <button onclick="window.location.href='?next=yes'"
          class="w-full rounded-md bg-green-600 px-4 py-3 text-base font-semibold text-white">
        Next
      </button>
    </div>
  </footer>
</div>
</body>
</html>
"""

PREDICT_HTML = """
<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<link crossorigin="" href="https://fonts.gstatic.com/" rel="preconnect"/>
<link as="style" href="https://fonts.googleapis.com/css2?display=swap&family=Noto+Sans:wght@400;500;700;900&family=Work+Sans:wght@400;500;700;900" onload="this.rel='stylesheet'" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet"/>
<title>Crop Recommendation</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<style type="text/tailwindcss">
    :root {
      --primary-color: #3ed411;
      --text-primary: #131811;
      --text-secondary: #6B7280;
      --background-primary: #F9FAFB;
      --background-secondary: #FFFFFF;
      --border-color: #E5E7EB;
    }
    .form-input {
      @apply w-full rounded-md border-gray-300 shadow-sm focus:border-[var(--primary-color)] focus:ring-[var(--primary-color)];
    }
    .form-label {
      @apply block text-sm font-medium text-gray-700 mb-1;
    }
    input[type="range"]::-webkit-slider-thumb {
      @apply w-5 h-5 bg-[var(--primary-color)] rounded-full cursor-pointer appearance-none;
    }
    input[type="range"]::-moz-range-thumb {
      @apply w-5 h-5 bg-[var(--primary-color)] rounded-full cursor-pointer;
    }
</style>
<style>
  body {
    min-height: max(884px, 100dvh);
  }
</style>
</head>
<body class="bg-[var(--background-primary)]" style='font-family: "Work Sans", "Noto Sans", sans-serif;'>
<div class="relative flex h-auto min-h-screen w-full flex-col justify-between group/design-root overflow-x-hidden">
<main class="flex-grow">
<header class="bg-[var(--background-secondary)] shadow-sm sticky top-0 z-10">
<div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
<div class="flex h-16 items-center justify-between">
<div class="flex items-center">
<button onclick="window.location.href='?back_to_profile=yes'" class="rounded-md p-2 text-gray-500 hover:bg-gray-100 hover:text-gray-600">
<span class="material-symbols-outlined">
  arrow_back_ios_new
</span>
</button>
</div>
<h1 class="text-lg font-bold text-gray-900">Crop Recommendation</h1>
<div class="w-10"></div>
</div>
</div>
</header>
<div class="py-6">
<div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
<div class="mb-6">
<h2 class="text-2xl font-bold leading-tight tracking-tight text-gray-900">Enter Soil & Climate Data</h2>
<p class="mt-1 text-sm text-gray-600">Provide the following details to get a crop recommendation.</p>
</div>
<div class="space-y-6 bg-white p-6 rounded-lg shadow">
<div class="relative">
<div aria-hidden="true" class="absolute inset-0 flex items-center">
<div class="w-full border-t border-gray-300"></div>
</div>
<div class="relative flex justify-center">
<span class="bg-white px-2 text-sm text-gray-500">Or enter manually</span>
</div>
</div>
</div>
<form id="recommendation-form" class="space-y-8 mt-6">
<div class="bg-white p-6 rounded-lg shadow">
<h3 class="text-lg font-semibold leading-6 text-gray-900 border-b border-gray-200 pb-3 mb-6">Soil Data</h3>
<div class="space-y-6">
<div>
<label class="form-label" for="nitrogen">Nitrogen (N)</label>
<div class="relative">
<input class="form-input" id="nitrogen" name="nitrogen" placeholder="e.g., 90" type="number" step="any" required/>
<div class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
<span class="text-gray-500 sm:text-sm">kg/ha</span>
</div>
</div>
</div>
<div>
<label class="form-label" for="phosphorus">Phosphorus (P)</label>
<div class="relative">
<input class="form-input" id="phosphorus" name="phosphorus" placeholder="e.g., 42" type="number" step="any" required/>
<div class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
<span class="text-gray-500 sm:text-sm">kg/ha</span>
</div>
</div>
</div>
<div>
<label class="form-label" for="potassium">Potassium (K)</label>
<div class="relative">
<input class="form-input" id="potassium" name="potassium" placeholder="e.g., 43" type="number" step="any" required/>
<div class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
<span class="text-gray-500 sm:text-sm">kg/ha</span>
</div>
</div>
</div>
<div>
<label class="form-label" for="ph">pH</label>
<div class="flex items-center space-x-4">
<input class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" id="ph" max="14" min="0" name="ph" step="0.1" type="range" value="7.0" oninput="this.nextElementSibling.value=this.value"/>
<input class="form-input w-24 text-center" step="0.1" type="number" value="7.0" oninput="this.previousElementSibling.value=this.value"/>
</div>
</div>
</div>
</div>
<div class="bg-white p-6 rounded-lg shadow">
<h3 class="text-lg font-semibold leading-6 text-gray-900 border-b border-gray-200 pb-3 mb-6">Climate Data</h3>
<div class="space-y-6">
<div>
<label class="form-label" for="temperature">Temperature</label>
<div class="relative">
<input class="form-input" id="temperature" name="temperature" placeholder="e.g., 20.8" type="number" step="any" required/>
<div class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
<span class="text-gray-500 sm:text-sm">°C</span>
</div>
</div>
</div>
<div>
<label class="form-label" for="humidity">Humidity</label>
<div class="relative">
<input class="form-input" id="humidity" name="humidity" placeholder="e.g., 82" type="number" step="any" required/>
<div class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
<span class="text-gray-500 sm:text-sm">%</span>
</div>
</div>
</div>
<div>
<label class="form-label" for="rainfall">Rainfall</label>
<div class="relative">
<input class="form-input" id="rainfall" name="rainfall" placeholder="e.g., 202.9" type="number" step="any" required/>
<div class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
<span class="text-gray-500 sm:text-sm">mm</span>
</div>
</div>
</div>
</div>
</div>
<div>
<button class="flex w-full justify-center rounded-md border border-transparent bg-[var(--primary-color)] py-3 px-4 text-base font-bold text-white shadow-sm hover:bg-opacity-90 focus:outline-none focus:ring-2 focus:ring-[var(--primary-color)] focus:ring-offset-2" type="submit">
  Get Recommendation
</button>
</div>
</form>
</div>
</div>
</main>
<footer class="sticky bottom-0 bg-[var(--background-secondary)] border-t border-gray-200">
<div class="mx-auto flex max-w-7xl justify-around px-4 py-2 sm:px-6 lg:px-8">
<a class="flex flex-col items-center gap-1 p-2 rounded-md text-[var(--primary-color)]" href="#">
<span class="material-symbols-outlined">home</span>
<span class="text-xs font-medium">Home</span>
</a>
<a class="flex flex-col items-center gap-1 p-2 rounded-md text-gray-500 hover:text-[var(--primary-color)]" href="#">
<span class="material-symbols-outlined">recommend</span>
<span class="text-xs font-medium">Recommendations</span>
</a>
<a class="flex flex-col items-center gap-1 p-2 rounded-md text-gray-500 hover:text-[var(--primary-color)]" href="#">
<span class="material-symbols-outlined">person</span>
<span class="text-xs font-medium">Profile</span>
</a>
<a class="flex flex-col items-center gap-1 p-2 rounded-md text-gray-500 hover:text-[var(--primary-color)]" href="#">
<span class="material-symbols-outlined">settings</span>
<span class="text-xs font-medium">Settings</span>
</a>
</div>
</footer>
</div>
<script>
    // This script handles form submission and passes data back to Streamlit
    document.getElementById('recommendation-form').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission

        const form = event.target;
        const formData = new FormData(form);
        const params = new URLSearchParams();

        for (const [key, value] of formData.entries()) {
            params.append(key, value);
        }

        // Redirect to a new URL with the form data as query parameters
        window.location.href = ?page=result&${params.toString()};
    });

    // Handle pH slider and number input synchronization
    const phRange = document.getElementById('ph');
    const phNumber = phRange.nextElementSibling;
    phRange.oninput = () => { phNumber.value = phRange.value; };
    phNumber.oninput = () => { phRange.value = phNumber.value; };
</script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <link crossorigin="" href="https://fonts.gstatic.com/" rel="preconnect"/>
    <link as="style" href="https://fonts.googleapis.com/css2?display=swap&family=Noto+Sans:wght@400;500;700;900&family=Work+Sans:wght@400;500;700;900" onload="this.rel='stylesheet'" rel="stylesheet"/>
    <title>Crop Recommendation Result</title>
    <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
    <style type="text/tailwindcss">
      :root {
        --primary-500: #22c55e;
        --primary-600: #16a34a;
        --primary-700: #15803d;
      }
      body { min-height: max(884px, 100dvh); font-family: "Work Sans", "Noto Sans", sans-serif; }
    </style>
</head>
<body class="bg-gray-50">
<div class="relative flex h-auto min-h-screen w-full flex-col bg-white">
    <header class="sticky top-0 z-10 bg-white border-b border-gray-200">
        <div class="mx-auto flex h-16 max-w-md items-center justify-between px-4">
            <h1 class="text-lg font-bold text-gray-900">Recommendation Result</h1>
        </div>
    </header>

    <main class="flex-1 overflow-y-auto">
        <div class="mx-auto max-w-md p-4 flex flex-col items-center justify-center h-full">
            <div class="text-center space-y-6">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-24 w-24 text-green-500 mx-auto" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                </svg>
                <h2 class="text-2xl font-bold text-green-700">Recommended Crop</h2>
                <div class="bg-green-100 p-6 rounded-xl shadow-lg">
                    <p class="text-5xl font-extrabold text-green-800" id="result-text">---</p>
                </div>
                <p class="text-sm text-gray-500">
                    This recommendation is based on the data you provided. The model suggests this crop as the most suitable for your specific conditions.
                </p>
            </div>
        </div>
    </main>

    <footer class="sticky bottom-0 bg-white border-t border-gray-200">
        <div class="mx-auto max-w-md p-4 flex items-center space-x-4">
            <button onclick="window.location.href='?back_to_predict=yes'" class="w-full rounded-md border border-gray-300 bg-white px-4 py-3 text-base font-semibold text-gray-700">
                Start Again
            </button>
            <button onclick="window.location.href='?go_to_dashboard=yes'" class="w-full rounded-md bg-green-600 px-4 py-3 text-base font-semibold text-white">
                View Dashboard
            </button>
        </div>
    </footer>
</div>
<script>
    // Get the query parameters to read the prediction
    const urlParams = new URLSearchParams(window.location.search);
    const prediction = urlParams.get('prediction');
    if (prediction) {
        document.getElementById('result-text').innerText = prediction.toUpperCase();
    }
</script>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<link crossorigin="" href="https://fonts.gstatic.com/" rel="preconnect"/>
<link as="style" href="https://fonts.googleapis.com/css2?display=swap&amp;family=Noto+Sans%3Awght%40400%3B500%3B700%3B900&amp;family=Work+Sans%3Awght%40400%3B500%3B700%3B900" onload="this.rel='stylesheet'" rel="stylesheet"/>
<title>Crop Recommendation Dashboard</title>
<link href="data:image/x-icon;base64," rel="icon" type="image/x-icon"/>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet"/>
<style type="text/tailwindcss">
      :root {
        --primary-color: #3fd411;
        --primary-50: #f0fdf4;
        --primary-100: #dcfce7;
        --primary-200: #bbf7d0;
        --primary-300: #86efac;
        --primary-400: #4ade80;
        --primary-500: #22c55e;
        --primary-600: #16a34a;
        --primary-700: #15803d;
        --primary-800: #166534;
        --primary-900: #14532d;
        --primary-950: #052e16;
      }
      body {
        font-family: "Work Sans", "Noto Sans", sans-serif;
      }
    </style>
<style>
    body {
      min-height: max(884px, 100dvh);
    }
  </style>
<style>
    body {
      min-height: max(884px, 100dvh);
    }
  </style>
  </head>
<body class="bg-gray-50">
<div class="relative flex h-auto min-h-screen w-full flex-col justify-between group/design-root overflow-x-hidden" style='font-family: "Work Sans", "Noto Sans", sans-serif;'>
<div class="flex-grow">
<header class="flex items-center justify-between bg-white p-4 shadow-sm">
<button onclick="window.location.href='?back_to_result=yes'" class="text-gray-700">
<span class="material-symbols-outlined"> arrow_back </span>
</button>
<h1 class="text-lg font-bold text-gray-800">Crop Recommendation Dashboard</h1>
<button class="text-gray-700">
<span class="material-symbols-outlined"> more_vert </span>
</button>
</header>
<main class="p-4 md:p-6">
<section class="mb-6">
<div class="flex justify-between items-center">
<h2 class="text-2xl font-bold text-gray-800">Your Dashboard</h2>
<button class="flex items-center gap-2 text-sm font-medium text-primary-600">
<span>Customize</span>
<span class="material-symbols-outlined text-base"> edit </span>
</button>
</div>
<p class="mt-1 text-gray-600">Dynamic data for informed farming decisions.</p>
</section>
<section class="mb-6">
<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
<div class="flex items-center justify-between mb-4">
<h3 class="font-bold text-gray-800">Real-Time Weather</h3>
<span class="text-sm text-gray-500">Sunny, 28°C</span>
</div>
<div class="flex justify-around">
<div class="text-center">
<p class="text-xs text-gray-500">Now</p>
<span class="material-symbols-outlined text-yellow-500 text-3xl">wb_sunny</span>
<p class="font-bold">28°</p>
</div>
<div class="text-center">
<p class="text-xs text-gray-500">10 AM</p>
<span class="material-symbols-outlined text-yellow-500 text-3xl">wb_sunny</span>
<p class="font-bold">29°</p>
</div>
<div class="text-center">
<p class="text-xs text-gray-500">1 PM</p>
<span class="material-symbols-outlined text-gray-400 text-3xl">cloud</span>
<p class="font-bold">31°</p>
</div>
<div class="text-center">
<p class="text-xs text-gray-500">4 PM</p>
<span class="material-symbols-outlined text-blue-400 text-3xl">water_drop</span>
<p class="font-bold">27°</p>
</div>
</div>
</div>
</section>
<section class="mb-6">
<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
<h3 class="font-bold text-gray-800 mb-4">7-Day Forecast</h3>
<div class="space-y-3">
<div class="flex justify-between items-center">
<p class="font-medium text-gray-600 w-1/4">Today</p>
<div class="flex items-center gap-2 w-1/4">
<span class="material-symbols-outlined text-yellow-500">wb_sunny</span>
<p class="text-sm">Sunny</p>
</div>
<p class="text-right w-1/4">32°/22°</p>
<p class="text-right w-1/4 text-blue-500">10%</p>
</div>
<div class="flex justify-between items-center">
<p class="font-medium text-gray-600 w-1/4">Mon</p>
<div class="flex items-center gap-2 w-1/4">
<span class="material-symbols-outlined text-gray-400">cloud</span>
<p class="text-sm">Cloudy</p>
</div>
<p class="text-right w-1/4">29°/21°</p>
<p class="text-right w-1/4 text-blue-500">20%</p>
</div>
<div class="flex justify-between items-center">
<p class="font-medium text-gray-600 w-1/4">Tue</p>
<div class="flex items-center gap-2 w-1/4">
<span class="material-symbols-outlined text-blue-400">water_drop</span>
<p class="text-sm">Rain</p>
</div>
<p class="text-right w-1/4">27°/20°</p>
<p class="text-right w-1/4 text-blue-500">80%</p>
</div>
</div>
</div>
</section>
<section class="mb-6">
<h3 class="text-xl font-bold text-gray-800 mb-4">Predictive Analytics</h3>
<div class="grid grid-cols-2 gap-4">
<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
<p class="text-sm font-medium text-gray-500">Soybean Yield</p>
<p class="text-2xl font-bold text-gray-800">2.7 <span class="text-lg">tons/acre</span></p>
<p class="text-xs text-green-600 mt-1">+0.2 vs avg</p>
</div>
<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
<p class="text-sm font-medium text-gray-500">Corn Yield</p>
<p class="text-2xl font-bold text-gray-800">3.1 <span class="text-lg">tons/acre</span></p>
<p class="text-xs text-green-600 mt-1">+0.4 vs avg</p>
</div>
</div>
</section>
<section class="mb-6">
<h3 class="text-xl font-bold text-gray-800 mb-4">Compare Recommendations</h3>
<div class="flex gap-4 overflow-x-auto pb-2 -mx-4 px-4">
<div class="flex-shrink-0 w-3/4 rounded-lg border border-primary-500 bg-primary-50 p-4 shadow-sm">
<p class="font-bold text-lg text-primary-700">Soybeans (Rec.)</p>
<p class="text-sm text-gray-600 mt-1">Est. Profit: <span class="font-medium">$350/acre</span></p>
<p class="text-sm text-gray-600">Planting: <span class="font-medium">May 1-15</span></p>
<p class="text-sm text-gray-600">Harvest: <span class="font-medium">Oct 1-15</span></p>
</div>
<div class="flex-shrink-0 w-3/4 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
<p class="font-bold text-lg text-gray-800">Corn</p>
<p class="text-sm text-gray-600 mt-1">Est. Profit: <span class="font-medium">$320/acre</span></p>
<p class="text-sm text-gray-600">Planting: <span class="font-medium">Apr 15-30</span></p>
<p class="text-sm text-gray-600">Harvest: <span class="font-medium">Sep 15-30</span></p>
</div>
<div class="flex-shrink-0 w-3/4 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
<p class="font-bold text-lg text-gray-800">Wheat</p>
<p class="text-sm text-gray-600 mt-1">Est. Profit: <span class="font-medium">$280/acre</span></p>
<p class="text-sm text-gray-600">Planting: <span class="font-medium">Sep 1-15</span></p>
<p class="text-sm text-gray-600">Harvest: <span class="font-medium">Jun 1-15</span></p>
</div>
</div>
</section>
<section class="mb-6">
<h3 class="text-xl font-bold text-gray-800 mb-4">Market Prices</h3>
<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
<div class="h-40">
<svg fill="none" height="100%" preserveAspectRatio="none" viewBox="0 0 472 150" width="100%" xmlns="http://www.w3.org/2000/svg">
<path d="M0 80C18.1538 80 18.1538 60 36.3077 60C54.4615 60 54.4615 90 72.6154 90C90.7692 90 90.7692 40 108.923 40C127.077 40 127.077 75 145.231 75C163.385 75 163.385 55 181.538 55C199.692 55 199.692 25 217.846 25C236 25 236 65 254.154 65C272.308 65 272.308 100 290.462 100C308.615 100 308.615 120 326.769 120C344.923 120 344.923 80 363.077 80C381.231 80 381.231 40 399.385 40C417.538 40 417.538 90 435.692 90C453.846 90 453.846 60 472 60" stroke="#fb923c" stroke-linecap="round" stroke-width="3"></path>
<path d="M0 109C18.1538 109 18.1538 21 36.3077 21C54.4615 21 54.4615 41 72.6154 41C90.7692 41 90.7692 93 108.923 93C127.077 93 127.077 33 145.231 33C163.385 33 163.385 101 181.538 101C199.692 101 199.692 61 217.846 61C236 61 236 45 254.154 45C272.308 45 272.308 121 290.462 121C308.615 121 308.615 149 326.769 149C344.923 149 344.923 1 363.077 1C381.231 1 381.231 81 399.385 81C417.538 81 417.538 129 435.692 129C453.846 129 453.846 25 472 25" stroke="var(--primary-400)" stroke-linecap="round" stroke-width="3"></path>
</svg>
</div>
<div class="flex justify-between items-center mt-2 border-t border-gray-200 pt-2">
<div class="flex items-center gap-2">
<div class="w-3 h-3 rounded-full bg-primary-400"></div>
<span class="text-sm font-medium text-gray-600">Soybeans</span>
</div>
<div class="flex items-center gap-2">
<div class="w-3 h-3 rounded-full bg-orange-400"></div>
<span class="text-sm font-medium text-gray-600">Corn</span>
</div>
</div>
</div>
</section>
</main>
</div>
<footer class="sticky bottom-0 border-t border-gray-200 bg-white shadow-t-sm">
<nav class="flex justify-around px-2 py-1">
<a class="flex flex-1 flex-col items-center justify-center gap-1 py-2 text-gray-600 hover:text-primary-600" href="#">
<span class="material-symbols-outlined"> home </span>
<p class="text-xs font-medium">Home</p>
</a>
<a class="flex flex-1 flex-col items-center justify-center gap-1 rounded-md bg-primary-100 py-2 text-primary-600" href="#">
<span class="material-symbols-outlined">bar_chart</span>
<p class="text-xs font-medium">Dashboard</p>
</a>
<a class="flex flex-1 flex-col items-center justify-center gap-1 py-2 text-gray-600 hover:text-primary-600" href="#">
<span class="material-symbols-outlined"> person </span>
<p class="text-xs font-medium">Profile</p>
</a>
<a class="flex flex-1 flex-col items-center justify-center gap-1 py-2 text-gray-600 hover:text-primary-600" href="#">
<span class="material-symbols-outlined"> settings </span>
<p class="text-xs font-medium">Settings</p>
</a>
</nav>
</footer>
</div>
</body></html>
"""

# Load ML model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error(f"Error: Could not find model or scaler files at '{MODEL_PATH}' or '{SCALER_PATH}'. Please check the paths.")
    st.stop()

# Session state to control navigation and pass data
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# --- PAGE 1: WELCOME SCREEN ---
if st.session_state.page == "welcome":
    components.html(WELCOME_HTML, height=500)
    query_params = st.query_params
    if "continue" in query_params:
        st.session_state.page = "profile"
        st.query_params.clear()

# --- PAGE 2: FARM PROFILE SETUP ---
elif st.session_state.page == "profile":
    components.html(PROFILE_HTML, height=700)
    query_params = st.query_params
    if "back" in query_params:
        st.session_state.page = "welcome"
        st.query_params.clear()
    elif "next" in query_params:
        st.session_state.page = "predict"
        st.query_params.clear()

# --- PAGE 3: PREDICTION FORM (New HTML) ---
elif st.session_state.page == "predict":
    query_params = st.query_params
    
    # Check if form data is present in the URL
    if all(key in query_params for key in ['nitrogen', 'phosphorus', 'potassium', 'ph', 'temperature', 'humidity', 'rainfall']):
        try:
            # Extract and convert data from URL parameters
            N = float(query_params['nitrogen'])
            P = float(query_params['phosphorus'])
            K = float(query_params['potassium'])
            ph = float(query_params['ph'])
            temperature = float(query_params['temperature'])
            humidity = float(query_params['humidity'])
            rainfall = float(query_params['rainfall'])
            
            # Process and predict
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled = scaler.transform(features)
            prediction = model.predict(scaled)[0]
            
            # Store prediction and navigate to result page
            st.session_state.prediction = prediction
            st.session_state.page = "result"
            st.query_params.clear()
            st.query_params['prediction'] = prediction
            
        except (ValueError, IndexError) as e:
            st.error(f"Error processing input data: {e}")
            st.write("Please go back and ensure all fields are filled with valid numbers.")
            components.html(PREDICT_HTML, height=1000) # Re-render the form
    else:
        # If no data is in the URL, show the form
        components.html(PREDICT_HTML, height=1000)
        if "back_to_profile" in query_params:
            st.session_state.page = "profile"
            st.query_params.clear()
        
# --- PAGE 4: RESULT PAGE ---
elif st.session_state.page == "result":
    components.html(RESULT_HTML, height=700)
    query_params = st.query_params
    if "back_to_predict" in query_params:
        st.session_state.page = "predict"
        st.query_params.clear()
    elif "go_to_dashboard" in query_params:
        st.session_state.page = "dashboard"
        st.query_params.clear()

# --- PAGE 5: DASHBOARD PAGE ---
elif st.session_state.page == "dashboard":
    components.html(DASHBOARD_HTML, height=1000)
    query_params = st.query_params
    if "back_to_result" in query_params:
        st.session_state.page = "result"
        st.query_params.clear()