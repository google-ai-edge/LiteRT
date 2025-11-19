# LiteRT.js Image Upscaler Demo

This demo uses LiteRT.js and the [Real-ESRGAN x4plus
model](https://github.com/xinntao/Real-ESRGAN) to upscale an image in the
browser

## How to Run

1.  **Build the monorepo:**

    See instructions in the [LiteRT.js readme](https://github.com/google-ai-edge/LiteRT/blob/main/litert/js/README.md).

2.  **Start the Development Server:**

    ```bash
    npm run dev
    ```

    This will start a local development server and provide a URL to access the
    application.

3.  **Build for Deployment:**

    ```bash
    npm run build
    ```

    This will create a `dist` directory with the bundled application.

## How It Works

When an image is uploaded, it is divided into smaller tiles. Each tile is then
processed by the selected ESRGAN model, which is executed by the LiteRT.js
runtime. The upscaled tiles are then stitched back together to form the final
high-resolution image.

## Models

The following models are used in this demo:

*   [**Real-ESRGAN
    x4plus:**](https://huggingface.co/qualcomm/Real-ESRGAN-x4plus) A 4x
    upscaling model for general images, converted to LiteRT by Qualcomm.

## File Structure

*   `index.html`: The main HTML file that loads the application.
*   `src/image_upscaler.ts`: The main Lit component that defines the UI and
    handles user interactions.
*   `src/upscaler.ts`: The core logic for upscaling the image, including the
    tiling strategy.
*   `src/styles.ts`: The CSS styles for the Lit component.
