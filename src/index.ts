import useBodyPix from "./core/hooks/useBodyPix";
import useRenderingPipeline, { renderCallbackType } from "./core/hooks/useRenderingPipeline"
import useTFLite from "./core/hooks/useTFLite"
import * as tfBodyPix from '@tensorflow-models/body-pix'

import { SourcePlayback } from './core/helpers/sourceHelper'

export { useBodyPix, useRenderingPipeline, useTFLite };

import {
  BackgroundConfig,
  backgroundImageUrls,
} from './core/helpers/backgroundHelper'
import { PostProcessingConfig } from './core/helpers/postProcessingHelper'
import { SegmentationConfig } from './core/helpers/segmentationHelper'

import { TFLite } from "./core/hooks/useTFLite"

const initVirtualBackground = () => {

  let sourcePlayback: SourcePlayback;
  const setSourcePlayback = (config: SourcePlayback) => {
    sourcePlayback = config
  }


  let backgroundConfig: BackgroundConfig;

  const setBackgroundConfig = (config: BackgroundConfig) => {
    backgroundConfig = config
  }

  let segmentationConfig: SegmentationConfig
  const setSegmentationConfig = (config: SegmentationConfig) => {
    segmentationConfig = config
  }

  let postProcessingConfig: PostProcessingConfig = {
    smoothSegmentationMask: true,
    jointBilateralFilter: { sigmaSpace: 1, sigmaColor: 0.1 },
    coverage: [0.5, 0.75],
    lightWrapping: 0.3,
    blendMode: 'screen',
  }

  const setPostProcessingConfig = (config: PostProcessingConfig) => {
    postProcessingConfig = config
  }

  let bodyPix: tfBodyPix.BodyPix;
  let tflite: TFLite;

  let cleanUpBodyPix: any;
  let cleanUpPipeline: any;

  const init = async () => {
    const { init, cleanup } = useBodyPix()
    cleanUpBodyPix = cleanup
    bodyPix = await init()

    const { loadMeetModel } = useTFLite()
    tflite = await loadMeetModel(segmentationConfig) as TFLite


    console.log("bodyPix", bodyPix)
    console.log("tflite", tflite)
  }

  const render = async (canvasRef: HTMLCanvasElement, callback: renderCallbackType) => {
    console.log("render virtual background", sourcePlayback,
      backgroundConfig,
      segmentationConfig)
    if (!sourcePlayback.htmlElement)
      return

    if (!segmentationConfig) {
      return
    }



    const {
      render,
      cleanup
    } = useRenderingPipeline(
      sourcePlayback,
      backgroundConfig,
      segmentationConfig,
      bodyPix,
      tflite
    )

    cleanUpPipeline = cleanup

    const {
      pipeline
    } = await render(canvasRef, callback)

    if (pipeline) {
      pipeline.updatePostProcessingConfig(postProcessingConfig)
    }
  }

  function cleanup() {
    cleanUpPipeline && cleanUpPipeline()
    cleanUpBodyPix && cleanUpBodyPix()
  }


  function enableMeet() {
    let segConfig: SegmentationConfig = {
      model: 'meet',
      backend: 'wasm',
      inputResolution: '96p',
      pipeline: 'webgl2',
    }
    setSegmentationConfig(segConfig)
  }
  function enableBodyPix() {
    let segConfig: SegmentationConfig = {
      model: "bodyPix",
      backend: "webgl",
      inputResolution: "360p",
      pipeline: "canvas2dCpu"
    }
    setSegmentationConfig(segConfig)
  }

  return {
    init,
    render,
    cleanup,
    enableBodyPix,
    enableMeet,
    setBackgroundConfig,
    setSourcePlayback
  }
}

export default initVirtualBackground