/**
  ******************************************************************************
  * @file    mt25tl01g_conf_template.h
  * @author  MCD Application Team
  * @brief   This file contains all the description of the
  *          MT25TL01G QSPI memory.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2019 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef MT25TL01G_CONF_H
#define MT25TL01G_CONF_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx.h"      // Changed from stm32xxxx.h
#include "stm32h7xx_hal.h"  // Changed from stm32xxxx_hal.h
/** @addtogroup BSP
  * @{
  */

#define CONF_MT25TL01G_READ_ENHANCE           0        // MMP performance enhance read — keep 0 for stability

#define CONF_QSPI_ODS                         MT25TL01G_CR_ODS_15  // Output driver strength — 15 ohms, default

#define CONF_QSPI_DUMMY_CLOCK                 8U       // General dummy clock cycles

/* Dummy cycles for STR (Single Transfer Rate) read mode */
#define MT25TL01G_DUMMY_CYCLES_READ_QUAD      10U       // Quad SPI read dummy cycles
#define MT25TL01G_DUMMY_CYCLES_READ           10U       // Single SPI read dummy cycles

/* Dummy cycles for DTR (Double Transfer Rate) read mode */
#define MT25TL01G_DUMMY_CYCLES_READ_DTR       6U       // DTR read dummy cycles
#define MT25TL01G_DUMMY_CYCLES_READ_QUAD_DTR  8U       // Quad DTR read dummy cycles
#ifdef __cplusplus
}
#endif

#endif /* MT25TL01G_CONF_H */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
