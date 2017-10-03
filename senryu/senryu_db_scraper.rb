require 'logger'
require 'mechanize'
require 'csv'
require './page_saver.rb'

############## Initialize
HTML_OUT_DIR = File.join('file',"html")
CSV_OUT_DIR = File.join('file',"csv")
$log = Logger.new(STDOUT)
$log.level = Logger::INFO
#$log.level = Logger::DEBUG
log = $log

$log.info {"#{File.basename(__FILE__)} start"}

$agent = Mechanize.new
$agent.user_agent_alias = 'Windows IE 10'
$agent.log = $log
html_saver = PageSaver.new(outdir: HTML_OUT_DIR, log: log)
html_seq = 0


def get_page(url)
  begin
    return $agent.get(url)
    return page
  rescue Timeout::Error => e
    $log.warn {e}
    $log.warn {"timeout occured,retry"}
    retry
  rescue => e
    $log.fatal {"Exception Occured at get url !"}
    $log.fatal {e}
    $log.info {"#{File.basename(__FILE__)} abnormal end"}
    exit 9
  end
end
$log.info "Getting top page"

url = "http://www.okajoki.com/db/search.php?page="

page = get_page("#{url}1")
html_saver.save(page,:filename=>"senryudb_1.html")

total_pages = page.at('div.maxpage > strong')&.text&.to_i
log.info {"total pages = #{total_pages}"}

CSV.open("#{CSV_OUT_DIR}/senryudb.csv","w") do |csv|
  (2..total_pages).each do |page_num|
    sakuhins = page.search('td.search_sakuhin')
    sakuhins.each do |s|
      eval = s.parent.next&.next&.at('div.search_desc')
#      log.debug {"#{s.text},#{eval.text.match(/［.*］/)}"}
      csv << [s.text.strip, eval&.text&.match(/［.*］/)&.to_s&.strip]
    end
    log.info "Getting page #{page_num}"
    page = get_page("#{url}#{page_num}")
    html_saver.save(page,:filename=>"senryudb_#{page_num}.html")
#    break if page_num > 5
  end
end

$log.info {"#{File.basename(__FILE__)} normal end"}
exit

$log.info {"#{File.basename(__FILE__)} normal end"}
