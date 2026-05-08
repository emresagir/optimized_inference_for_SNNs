/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "eth.h"
#include "fdcan.h"
#include "ltdc.h"
#include "quadspi.h"
#include "rtc.h"
#include "sai.h"
#include "usart.h"
#include "usb_otg.h"
#include "gpio.h"
#include "fmc.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "test.h"
#include "lif_neuron_gen.h"
#include "stm32h750b_discovery_qspi.h"
#include "mt25tl01g.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#ifndef HSEM_ID_0
#define HSEM_ID_0 (0U) /* HW semaphore 0*/
#endif

#define TEST_OUTPUT 20
#define NUM_STEPS 256 //number of timesteps to simulate
#define INPUTS 12
#define OUTPUTS 7

extern QSPI_HandleTypeDef hqspi;


/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* accumulate_output holds integer spike counts for each output neuron */
int32_t accumulate_output[OUTPUTS];


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* ── Helpers ────────────────────────────────────────────────────────────── */

/**
 * tiny_itoa – write a non-negative integer into buf and return the
 * number of characters written (no null terminator).
 */
int tiny_itoa(uint32_t v, char *buf) {
    if (v == 0) { buf[0] = '0'; return 1; }
    char tmp[12];
    int  len = 0;
    while (v) { tmp[len++] = '0' + (v % 10); v /= 10; }
    for (int i = 0; i < len; i++) buf[i] = tmp[len - 1 - i];
    return len;
}

/* ── Core classification logic ──────────────────────────────────────────── */

/**
 * classify_sample
 * ───────────────
 * Feeds one test sample (all its timesteps) through the SNN and returns
 * the predicted class (0-9) together with per-class spike totals.
 *
 * @param sample          Pointer to the TestSample descriptor.
 * @param spike_totals    Output array [TD_NUM_CLASSES] – caller provides storage.
 * @return                Predicted class index (argmax of spike_totals).
 */
uint8_t classify_sample(const TestSample *sample,
                                uint32_t         spike_totals[TD_NUM_CLASSES]) {
    static int8_t output_spikes[TD_NUM_CLASSES];

    /* Zero the accumulator */
    memset(spike_totals, 0, TD_NUM_CLASSES * sizeof(uint32_t));

    /* Reset neuron membrane potentials before the new gesture */
    SNN_Reset_State();

    /* Feed every timestep */
    for (uint16_t t = 0; t < sample->num_timesteps; t++) {

      //TEST 
        // char bufferotso [50];
        // snprintf(bufferotso, sizeof(bufferotso), "timestep = %d, \t\n", t);
        // usart1_print(bufferotso);

        SNN_Run_Timestep(sample->spikes[t], output_spikes);

        /* Accumulate spikes – output_spikes[c] is 0 or 1 */
        for (uint8_t c = 0; c < TD_NUM_CLASSES; c++) {
            if (output_spikes[c] > 0) {
                spike_totals[c]++;
            }
        }
    }

    /* Argmax: find the class with the most spikes */
    uint8_t  best_class = 0;
    uint32_t best_count = spike_totals[0];
    for (uint8_t c = 1; c < TD_NUM_CLASSES; c++) {
        if (spike_totals[c] > best_count) {
            best_count = spike_totals[c];
            best_class = c;
        }
    }
    return best_class;
}

/* ── Public entry point ─────────────────────────────────────────────────── */

/**
 * SNN_Run_Tests
 * ─────────────
 * Call once from main() after all peripherals are initialised.
 *
 * Example USART1 output (115200 baud):
 *
 *   ╔══════════════════════════════════╗
 *   ║  SNN ST-MNIST Inference Test     ║
 *   ║  Samples: 50   Classes: 10       ║
 *   ╚══════════════════════════════════╝
 *   [00] T=62  GT=8  Pred=8  PASS  spikes: 0 0 0 0 0 0 0 0 14 0
 *   [01] T=47  GT=3  Pred=3  PASS  spikes: 0 0 0 7 0 0 0 0 0  0
 *   ...
 *   ─────────────────────────────────
 *   Accuracy: 47/50  (94.00%)
 *   ─────────────────────────────────
 */
void SNN_Run_Tests(void) {
    /* ── Banner ── */
    usart1_print("\r\n");
    usart1_print("╔══════════════════════════════════════╗\r\n");
    usart1_print("║  SNN ST-MNIST Inference Test          ║\r\n");
    usart1_print("╚══════════════════════════════════════╝\r\n");

    {
        char buf[64];
        int  n;
        n  = 0;
        n += snprintf(buf + n, sizeof(buf) - n, "  Samples : %u\r\n", (unsigned)TD_NUM_SAMPLES);
        usart1_print(buf);
        n  = 0;
        n += snprintf(buf + n, sizeof(buf) - n, "  Classes : %u\r\n", (unsigned)TD_NUM_CLASSES);
        usart1_print(buf);
    }
    usart1_print("\r\n");

    /* Initialise SNN weights + neuron parameters once */
    SNN_Init();

    uint32_t correct = 0;
    uint32_t spike_totals[TD_NUM_CLASSES];

    for (uint16_t s = 0; s < TD_NUM_SAMPLES; s++) {
        const TestSample *smp = &test_samples[s];

        uint8_t predicted = classify_sample(smp, spike_totals);
        uint8_t gt        = smp->label;
        uint8_t pass      = (predicted == gt);
        if (pass) correct++;

        /* ── Per-sample line ─────────────────────────────────────────── */
        char line[160];
        int  pos = 0;

        /* Index */
        pos += snprintf(line + pos, sizeof(line) - pos, "[%02u] ", (unsigned)s);

        /* Timesteps */
        pos += snprintf(line + pos, sizeof(line) - pos, "T=%3u  ", (unsigned)smp->num_timesteps);

        /* Ground-truth / prediction / pass-fail */
        pos += snprintf(line + pos, sizeof(line) - pos,
                        "GT=%u  Pred=%u  %s",
                        (unsigned)gt,
                        (unsigned)predicted,
                        pass ? "PASS" : "FAIL");

        /* Spike totals per class */
        pos += snprintf(line + pos, sizeof(line) - pos, "  spikes:");
        for (uint8_t c = 0; c < TD_NUM_CLASSES; c++) {
            pos += snprintf(line + pos, sizeof(line) - pos,
                            " %lu", (unsigned long)spike_totals[c]);
        }

        pos += snprintf(line + pos, sizeof(line) - pos, "\r\n");
        usart1_print(line);
    }

    /* ── Summary ── */
    usart1_print("─────────────────────────────────────────\r\n");
    {
        char buf[80];
        /* Compute percentage with 2 decimal places without floating-point */
        uint32_t pct_int  = (correct * 100u) / TD_NUM_SAMPLES;
        uint32_t pct_frac = ((correct * 10000u) / TD_NUM_SAMPLES) % 100u;
        snprintf(buf, sizeof(buf),
                 "Accuracy: %lu/%u  (%lu.%02lu%%)\r\n",
                 (unsigned long)correct,
                 (unsigned)TD_NUM_SAMPLES,
                 (unsigned long)pct_int,
                 (unsigned long)pct_frac);
        usart1_print(buf);
    }
    usart1_print("─────────────────────────────────────────\r\n");
}



/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  // MPU_Config();
  // CPU_CACHE_Enable();

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();
  MX_ADC3_Init();
  MX_ETH_Init();
  MX_FDCAN1_Init();
  MX_FDCAN2_Init();
  MX_FMC_Init();
  MX_LTDC_Init();
  MX_QUADSPI_Init();
  MX_RTC_Init();
  MX_SAI2_Init();
  MX_USART1_UART_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  /* USER CODE BEGIN 2 */

  BSP_QSPI_EnableMemoryMappedMode(0);

  char msg[] = "U are sane, board works!\r\n";
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    HAL_UART_Transmit(&huart3, (uint8_t*)msg, strlen(msg), 100);
    HAL_GPIO_TogglePin(GPIOI, GPIO_PIN_13);  
    HAL_Delay(1000);
    HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_3);

    // char buf[32];
    // result = dsp_test();
    // int len = snprintf(buf, sizeof(buf), "Result= %d\r\n", (int) result);
    // HAL_UART_Transmit(&huart3, (uint8_t*)buf, len, 100);


    //test_cmsis_nn(&huart3); 

    SNN_Run_Tests();
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_LSI
                              |RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.LSIState = RCC_LSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 2;
  RCC_OscInitStruct.PLL.PLLN = 12;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 3;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOMEDIUM;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_ADC;
  PeriphClkInitStruct.PLL2.PLL2M = 2;
  PeriphClkInitStruct.PLL2.PLL2N = 12;
  PeriphClkInitStruct.PLL2.PLL2P = 2;
  PeriphClkInitStruct.PLL2.PLL2Q = 2;
  PeriphClkInitStruct.PLL2.PLL2R = 2;
  PeriphClkInitStruct.PLL2.PLL2RGE = RCC_PLL2VCIRANGE_3;
  PeriphClkInitStruct.PLL2.PLL2VCOSEL = RCC_PLL2VCOMEDIUM;
  PeriphClkInitStruct.PLL2.PLL2FRACN = 0;
  PeriphClkInitStruct.AdcClockSelection = RCC_ADCCLKSOURCE_PLL2;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
