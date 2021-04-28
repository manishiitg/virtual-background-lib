export type BackgroundConfig = {
  type: 'none' | 'blur' | 'image' | 'greenscreen'
  url?: string,
  width?: number,
  height?: number,
  htmlElement?: HTMLImageElement | HTMLVideoElement
}

export const backgroundImageUrls = [
  'architecture-5082700_1280',
  'porch-691330_1280',
  'saxon-switzerland-539418_1280',
  'shibuyasky-4768679_1280',
].map((imageName) => `${process.env.PUBLIC_URL}/backgrounds/${imageName}.jpg`)
