import axios from 'axios'

const configuredBaseUrl = import.meta.env.VITE_API_URL
const isViteDevServer = typeof window !== 'undefined' && window.location.port === '5173'
const baseURL = configuredBaseUrl || (isViteDevServer ? '/api' : 'http://127.0.0.1:8000')

const api = axios.create({
	baseURL,
	timeout: 10000,
})

export const getMetrics = () => api.get('/metrics').then(({ data }) => data)
export const getExceptions = () => api.get('/exceptions').then(({ data }) => data)
export const askQuestion = (question) => api.post('/ask', { question }).then(({ data }) => data)
export const runReconciliation = () => api.post('/reconcile').then(({ data }) => data)
export const generateDataset = (size) => api.post('/generate-data', { size }).then(({ data }) => data)
export const generateReport = () => api.post('/report').then(({ data }) => data)
