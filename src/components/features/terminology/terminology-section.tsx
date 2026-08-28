import React from 'react'
import { WEATHER_REFERENCES } from '@/lib/constants/weather-references'
import { getWarningStyles } from '@/lib/utils/weatherWarningUtil'

const TerminologySection = () => {
  const entries = Object.entries(WEATHER_REFERENCES)

  return (
    <section className="w-full max-w-7xl mx-auto mt-8 flex flex-col gap-10">
      {/* All metric references */}
      <div className="flex flex-col gap-10 w-full">
        {entries.map(([metricName, references]) => {
          if (!references || references.length === 0) return null

          return (
            <section key={metricName} className="w-full">
              <p className="border-l-4 md:text-lg text-base text-light border-l-main pl-2 mb-4 font-semibold">
                {metricName}
              </p>

              {/* Compact threshold bar */}
              <div className="grid grid-cols-2 md:flex md:flex-row mb-4">
                {references.map((ref, idx) => {
                  const warningStyles = getWarningStyles(ref.color as string | undefined)
                  const isLast = idx === references.length - 1
                  const isOddCount = references.length % 2 === 1
                  const shouldUseBlackText = 
                  ref.color === '#FBF300' || 
                  ref.color === '#E7F6FC' || 
                  ref.color === '#D6D6D6' ||
                  ref.color === '#FFFFE0' ||
                  ref.color === '#FFFF99' ||
                  ref.color === '#FFFACD'

                  return (
                    <div
                      key={ref.term ?? idx}
                      className='flex flex-col flex-1 md:justify-between mt-4 md:mt-0'
                    >

                      {/* Term + Definition */}
                      <div className="flex flex-col justify-start gap-1 flex-1">
                        <p className="text-sm md:text-md font-semibold text-left text-light">
                          {ref.term}
                          
                        </p>
                        <p className="text-xs md:text-xs text-left text-light font-light wrap-break-word">
                          {ref.definition}
                        </p>
                      </div>


                      {/* Thresholds */}
                      <div
                        className={[
                          'font-bold text-center border text-xs md:text-xs py-2 px-1 max-h-10 mt-4',
                          isLast && isOddCount ? 'col-span-2 md:col-span-1' : '',
                        ].join(' ')}
                        style={{
                          backgroundColor: ref.color || 'transparent',
                          borderColor: warningStyles.border,
                          color: shouldUseBlackText ? '#334155' : 'white',
                        }}
                      >
                        {ref.threshold}
                      </div>
                    </div>

                  )
                })}
              </div>

              
            </section>
          )
        })}
      </div>
    </section>
  )
}

export default TerminologySection