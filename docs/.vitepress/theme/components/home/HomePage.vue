<script setup lang="ts">
import CapabilityShowcase from "./CapabilityShowcase.vue"
import HomeInstall from "./HomeInstall.vue"
import HomeMultimodal from "./HomeMultimodal.vue"
import HomeResults from "./HomeResults.vue"
import HomeValues from "./HomeValues.vue"
import JetHero from "./JetHero.vue"
import "./home.css"
</script>

<template>
  <main class="jet-home">
    <JetHero />
    <CapabilityShowcase />
    <HomeValues />
    <HomeMultimodal />
    <HomeResults />
    <HomeInstall />
  </main>
</template>
