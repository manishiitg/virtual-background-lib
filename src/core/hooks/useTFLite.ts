// import { useEffect, useState } from 'react'
import { SegmentationConfig } from '../helpers/segmentationHelper'
import { simd } from "wasm-feature-detect"

declare function createTFLiteModule(): Promise<TFLite>
declare function createTFLiteSIMDModule(): Promise<TFLite>

export interface TFLite extends EmscriptenModule {
  _getModelBufferMemoryOffset(): number
  _getInputMemoryOffset(): number
  _getInputHeight(): number
  _getInputWidth(): number
  _getInputChannelCount(): number
  _getOutputMemoryOffset(): number
  _getOutputHeight(): number
  _getOutputWidth(): number
  _getOutputChannelCount(): number
  _loadModel(bufferSize: number): number
  _runInference(): number
}

export function useTFLite() {

  let tflite: TFLite;
  let tfliteSIMD: TFLite;
  let selectedTFLite: TFLite;
  let isSIMDSupported: boolean = false;

  //   const [tfliteSIMD, setTFLiteSIMD] = useState<TFLite>()
  //   const [selectedTFLite, setSelectedTFLite] = useState<TFLite>()
  //   const [isSIMDSupported, setSIMDSupported] = useState(false)

  async function load() {
    tflite = await createTFLiteModule()
    try {
      simd().then(async simdSupported => {
        if (simdSupported) {
          /* SIMD support */
          const createdTFLiteSIMD = await createTFLiteSIMDModule()
          tfliteSIMD = createdTFLiteSIMD
          isSIMDSupported = true
        } else {
          /* No SIMD support */
          isSIMDSupported = false
        }
      })

    } catch (error) {
      console.warn('Failed to create TFLite SIMD WebAssembly module.', error)
    }
    console.log("simd support", isSIMDSupported)
    return { tflite, tfliteSIMD, isSIMDSupported }
  }

  async function loadMeetModel(segmentationConfig: SegmentationConfig) {
    const { tflite, tfliteSIMD, isSIMDSupported } = await load()
    // segmentationConfig.model,
    // segmentationConfig.backend,
    // segmentationConfig.inputResolution

    if (isSIMDSupported) {
      if (segmentationConfig.backend === "wasm") {
        segmentationConfig.backend = "wasmSimd"
      }
    }

    if (
      !tflite ||
      (isSIMDSupported && !tfliteSIMD) ||
      (!isSIMDSupported && segmentationConfig.backend === 'wasmSimd') ||
      segmentationConfig.model !== 'meet'
    ) {
      return
    }

    selectedTFLite = null!

    const newSelectedTFLite =
      segmentationConfig.backend === 'wasmSimd' ? tfliteSIMD : tflite

    if (!newSelectedTFLite) {
      throw new Error(
        `TFLite backend unavailable: ${segmentationConfig.backend}`
      )
    }

    const modelFileName =
      segmentationConfig.inputResolution === '144p'
        ? 'segm_full_v679'
        : 'segm_lite_v681'
    console.log('Loading meet model:', modelFileName)

    const modelResponse = await fetch(
      `${process.env.PUBLIC_URL}/models/${modelFileName}.tflite`
    )
    const model = await modelResponse.arrayBuffer()
    console.log('Model buffer size:', model.byteLength)

    const modelBufferOffset = newSelectedTFLite._getModelBufferMemoryOffset()
    console.log('Model buffer memory offset:', modelBufferOffset)
    console.log('Loading model buffer...')
    newSelectedTFLite.HEAPU8.set(new Uint8Array(model), modelBufferOffset)
    console.log(
      '_loadModel result:',
      newSelectedTFLite._loadModel(model.byteLength)
    )

    console.log(
      'Input memory offset:',
      newSelectedTFLite._getInputMemoryOffset()
    )
    console.log('Input height:', newSelectedTFLite._getInputHeight())
    console.log('Input width:', newSelectedTFLite._getInputWidth())
    console.log('Input channels:', newSelectedTFLite._getInputChannelCount())

    console.log(
      'Output memory offset:',
      newSelectedTFLite._getOutputMemoryOffset()
    )
    console.log('Output height:', newSelectedTFLite._getOutputHeight())
    console.log('Output width:', newSelectedTFLite._getOutputWidth())
    console.log(
      'Output channels:',
      newSelectedTFLite._getOutputChannelCount()
    )

    selectedTFLite = newSelectedTFLite

    return selectedTFLite

  }

  return { loadMeetModel }
}


export default useTFLite
