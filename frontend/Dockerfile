FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY . .

# Set environment variables
ENV PORT=8501
ENV NODE_ENV=production

# Build the app
RUN npm run build

# Install serve to run the build
RUN npm install -g serve

# Expose port 8501
EXPOSE 8501

# Start the app
CMD ["serve", "-s", "build", "-l", "8501"]