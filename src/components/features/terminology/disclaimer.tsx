const Disclaimer = () => {
  return (
      <div className="mt-8 p-4 border bg-card text-muted-foreground backdrop-blur-md backdrop-brightness-110 border-l-4 border-l-yellow-400 text-sm rounded shadow-[0.0625rem_0.0625rem_0.9375rem_rgba(0,0,0,0.05)]">
        <strong>Disclaimer:</strong> Kloudtech does not claim any data taken from PAGASA for our
        knowledge base. If you need more information, visit their website at
        <a
          href="https://www.pagasa.dost.gov.ph/"
          target="_blank"
          className="text-main underline px-1"
        >
          pagasa.dost.gov.ph
        </a>
      </div>
  )
}

export default Disclaimer