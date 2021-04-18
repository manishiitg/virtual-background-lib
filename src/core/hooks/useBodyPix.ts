import * as tfBodyPix from '@tensorflow-models/body-pix'
import * as tf from '@tensorflow/tfjs'

export function useBodyPix() {
  let bodyPix: tfBodyPix.BodyPix

  async function init(): Promise<tfBodyPix.BodyPix> {
    if (bodyPix) {
      return bodyPix
    }
    console.log('Loading TensorFlow.js and BodyPix segmentation model')
    await tf.ready()
    bodyPix = await tfBodyPix.load()
    console.log('TensorFlow.js and BodyPix loaded')
    return bodyPix
  }

  function cleanup() {
    if (bodyPix) {
      bodyPix.dispose()
      bodyPix = null!
    }
  }

  return { init, cleanup }
}

export default useBodyPix
