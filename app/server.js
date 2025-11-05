import Fastify from 'fastify'
import path from 'path'
import { fileURLToPath } from 'url'
import fastifyStatic from '@fastify/static'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = Fastify()

app.register(fastifyStatic, {
  root: path.join(__dirname, 'public'),
})

app.listen({ port: 8080 }, (err, address) => {
  if (err) throw err
  console.log(`🚀 Serveur lancé sur ${address}`)
})
