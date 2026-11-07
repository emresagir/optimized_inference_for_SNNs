/**
  ******************************************************************************
  * @file    stm32h750b_discovery_conf_template.h
  * @author  MCD Application Team
  * @brief   STM32H750B-DK board configuration file.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2019 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef STM32H750B_DK_CONF_H
#define STM32H750B_DK_CONF_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"

/* COM define */
#define USE_COM_LOG                         0U  // No UART logging via BSP
#define USE_BSP_COM_FEATURE                 0U  // You handle UART yourself

/* Touch controllers defines */
#define USE_FT5336_TS_CTRL                  0U  // No touchscreen needed

#define LCD_LAYER_0_ADDRESS                 0xD0000000U  // Keep as-is (SDRAM address, harmless)
#define LCD_LAYER_1_ADDRESS                 0xD0200000U  // Keep as-is

/* Audio codecs defines */
#define USE_AUDIO_CODEC_WM8994              0U  // No audio needed

/* Default Audio IN internal buffer size */
#define DEFAULT_AUDIO_IN_BUFFER_SIZE        64U  // Keep as-is, harmless

/* TS supported features defines */
#define USE_TS_GESTURE                      0U  // No touchscreen
#define USE_TS_MULTI_TOUCH                  0U  // No touchscreen

/* Default TS touch number */
#define TS_TOUCH_NBR                        2U  // Keep as-is, harmless

/* IRQ priorities — keep all defaults, they don't affect QSPI */
#define BSP_SDRAM_IT_PRIORITY               15U
#define BSP_BUTTON_USER_IT_PRIORITY         15U
#define BSP_AUDIO_OUT_IT_PRIORITY           14U
#define BSP_AUDIO_IN_IT_PRIORITY            15U
#define BSP_SD_IT_PRIORITY                  14U
#define BSP_SD_RX_IT_PRIORITY               14U
#define BSP_SD_TX_IT_PRIORITY               15U
#define BSP_TS_IT_PRIORITY                  15U

#ifdef __cplusplus
}
#endif

#endif /* STM32H750B_DK_CONF_H */
