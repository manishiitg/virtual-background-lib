import { BodyPix } from '@tensorflow-models/body-pix'
import { buildCanvas2dPipeline } from '../../pipelines/canvas2d/canvas2dPipeline'
import { buildWebGL2Pipeline } from '../../pipelines/webgl2/webgl2Pipeline'
import { BackgroundConfig } from '../helpers/backgroundHelper'
import { RenderingPipeline } from '../helpers/renderingPipelineHelper'
import { SegmentationConfig } from '../helpers/segmentationHelper'
import { SourcePlayback } from '../helpers/sourceHelper'
import { TFLite } from './useTFLite'


import {
  CLEAR_TIMEOUT,
  TIMEOUT_TICK,
  SET_TIMEOUT,
  timerWorkerScript
} from './../helpers/TimeWorker';

import { WEBGL } from "../helpers/WebGL"

export interface renderCallbackType { (fps: number, durations: number[]): void }


export function useRenderingPipeline(
  sourcePlayback: SourcePlayback,
  backgroundConfig: BackgroundConfig,
  segmentationConfig: SegmentationConfig,
  bodyPix: BodyPix,
  tflite: TFLite
) {

  let pipeline: RenderingPipeline | null;
  let fps: number = 0
  let durations: number[] = [];
  let shouldRender = true
  let renderRequestId: number
  let newPipeline: any

  let webWorker = new Worker(timerWorkerScript, { name: 'Blur effect worker' });
  let screenOff = false

  async function render(canvasRef: HTMLCanvasElement, callback: renderCallbackType) {
    // The useEffect cleanup function is not enough to stop
    // the rendering loop when the framerate is low

    let previousTime = 0
    let beginTime = 0
    let eventCount = 0
    let frameCount = 0
    const frameDurations: number[] = []

    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === 'visible') {
        if (screenOff) {
          screenOff = false
          webWorker.postMessage({
            id: CLEAR_TIMEOUT
          });
        }
      } else {
        screenOff = true
        webWorker.postMessage({
          id: SET_TIMEOUT,
          timeMs: 1000 / 30
        });

      }
    });


    if (!WEBGL.isWebGLAvailable()) {
      segmentationConfig.pipeline === 'canvas2dCpu'
    }

    newPipeline =
      segmentationConfig.pipeline === 'webgl2'
        ? buildWebGL2Pipeline(
          sourcePlayback,
          backgroundConfig.htmlElement ? backgroundConfig.htmlElement as HTMLImageElement : null,
          backgroundConfig,
          segmentationConfig,
          canvasRef,
          tflite,
          addFrameEvent
        )
        : buildCanvas2dPipeline(
          sourcePlayback,
          backgroundConfig,
          segmentationConfig,
          canvasRef,
          bodyPix,
          tflite,
          addFrameEvent
        )

    async function render() {
      if (!shouldRender) {
        return
      }
      beginFrame()
      await newPipeline.render()
      endFrame()
      if (screenOff)
        webWorker.postMessage({
          id: SET_TIMEOUT,
          timeMs: 1000 / 30
        });
      else
        renderRequestId = requestAnimationFrame(render)
    }

    webWorker.onmessage = (response) => {
      if (response.data.id === TIMEOUT_TICK) {
        render()
      }
    }

    function beginFrame() {
      beginTime = Date.now()
    }

    function addFrameEvent() {
      const time = Date.now()
      frameDurations[eventCount] = time - beginTime
      beginTime = time
      eventCount++
    }

    function endFrame() {
      const time = Date.now()
      frameDurations[eventCount] = time - beginTime
      frameCount++
      if (time >= previousTime + 1000) {
        fps = (frameCount * 1000) / (time - previousTime)
        durations = frameDurations
        previousTime = time
        frameCount = 0
      }
      eventCount = 0
      callback(fps, durations)
    }

    render()
    console.log(
      'Animation started:',
      sourcePlayback,
      backgroundConfig,
      segmentationConfig
    )

    pipeline = newPipeline

    return {
      pipeline
    }
  }

  async function cleanup() {
    shouldRender = false
    cancelAnimationFrame(renderRequestId)
    newPipeline.cleanUp()
    webWorker.postMessage({
      id: CLEAR_TIMEOUT
    });

    webWorker.terminate();

    console.log(
      'Animation stopped:',
      sourcePlayback,
      backgroundConfig,
      segmentationConfig
    )

    pipeline = null
  }

  return {
    render,
    cleanup
  }

}

export default useRenderingPipeline
