<script setup lang="ts">
import CapabilityShowcase from "./CapabilityShowcase.vue"
import HomeInstall from "./HomeInstall.vue"
import HomeValues from "./HomeValues.vue"
import JetHero from "./JetHero.vue"
import "./home.css"
</script>

<template>
  <main class="jet-home">
    <JetHero />
    <CapabilityShowcase />
    <HomeValues />
    <HomeInstall />
  </main>
</template>
