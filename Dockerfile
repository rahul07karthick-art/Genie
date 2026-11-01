FROM node:22-alpine

WORKDIR /app

COPY package.json ./
COPY backend ./backend
COPY ml ./ml
COPY outputs ./outputs
COPY ml-artifacts ./ml-artifacts

ENV NODE_ENV=production
ENV NEERWATCH_READ_ONLY=true
ENV PORT=10000

EXPOSE 10000
USER node

CMD ["npm", "start"]
