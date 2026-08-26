import axios from 'axios'

const api = axios.create({ baseURL: import.meta.env.VITE_API_URL || '/api' })

export const getMetrics = () => api.get('/metrics').then(({ data }) => data)
export const getExceptions = () => api.get('/exceptions').then(({ data }) => data)
export const askQuestion = (question) => api.post('/ask', { question }).then(({ data }) => data)
export const runReconciliation = () => api.post('/reconcile').then(({ data }) => data)
export const generateReport = () => api.post('/report').then(({ data }) => data)
