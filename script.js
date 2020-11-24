    const header  	= document .querySelector( 'header'  );
    const summary 	= document .querySelector( 'summary' );
    let requestURL	= 'richard_cv.json'
    let request = new XMLHttpRequest();
    request .open( 'GET', requestURL );
    request .responseType = 'json' ;
    request .send() ;
    request .onload = function() {
      const superHero = request.response;
      populateHeader ( superHero );
      showWork		( superHero , 'Work life experience' ) ;
      showEducation	( superHero , 'Education' ) ;
      showCertificates	( superHero , 'Certificates' ) ;
      showLanguages	( superHero , 'Languages' ) ;
      showPublications	( superHero , 'Publications' ) ;      
      showOther		( superHero , 'Other Supporting Information' ) ;
      showFooter	( superHero , ' ' ) ;
    }

    function ThingyMahBob ( jsonOb , datablocktag, fieldtag , idtag ) {
	//
	// COLORS
	var fill = d3.scaleOrdinal().range(["#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4", "#E6F598", "#FFFFBF", "#FEE08B", "#FDAE61", "#F46D43", "#D53E4F", "#9E0142"]);
	//
	// WORDY BITS
	var myWords = jsonOb[datablocktag][0][fieldtag]
	for(let j = 1; j < jsonOb[datablocktag].length; j++) {
		myWords.concat( jsonOb[datablocktag][j][fieldtag] );   
	}
	//
	// SETTING SOME NUMBERS
	var margin = {top: 10, right: 10, bottom: 10, left: 10},
		width = 800 - margin.left - margin.right,
		height = 450 - margin.top - margin.bottom;
	//
	// PARSE AND PLAY WITH THE WORDS MAYBE , IDK ?
	function getWords(i) {
		return words[i]
            		.replace(/[!\.,:;\?]/g, '')
            		.split(' ')
            		//.map( function(d) {
            		//    return {text: d, size: 10 + Math.random() * 60};
            		//})
            		}
    }

    function populate ( hstr , ht , bstr , bt , jsonObj , qSelect , document ) {
      myThing = document .createElement( ht ) ;
      myThing .textContent = hstr + ':' ;
      qSelect .appendChild ( myThing ) ;
      myThing = document .createElement( bt ) ;
      myThing .textContent = jsonObj[bstr][hstr] ;
      qSelect .appendChild ( myThing ) ;
    }

    function populateHeader( jsonObj ) {
      const myH1 = document.createElement('h1') ;
      myH1.textContent = jsonObj['Personal']['Given name'] ;
      header.appendChild(myH1);

      populate ( 'Motto' , 'h4' , 'Personal' , 'p' , jsonObj, header, document )
      populate ( 'Birthday' , 'h4' , 'Personal' , 'p' , jsonObj, header, document )
      populate ( 'Primary skills' , 'h4' , 'Personal' , 'p' , jsonObj, header, document )      
      populate ( 'Stress character' , 'h4' , 'Personal' , 'p' , jsonObj, header, document )
      
      myContactInformation = document.createElement('h4') ;
      myContactInformation .textContent = 'Contact information:' ;
      header.appendChild( myContactInformation ) ;
      myContactInformation = document.createElement('p') ;
      myContactInformation .textContent = jsonObj['Personal']['Contact information']['Mobile'] +
       " | " + jsonObj['Personal']['Contact information']['E-mail'] ;
      header.appendChild( myContactInformation ) ;
      populate ( 'History' , 'h2' , 'Personal' , 'p' , jsonObj, header, document )
    }

    function showPublications( jsonObj , fieldtag ) {
      const section	= document .querySelector ( '#publications' );
      const myBlock	= document .createElement ( 'block' );

      Header	= document .createElement ( 'h2' ) ;
      Header	.textContent = fieldtag ;
      myBlock	.appendChild ( Header ) ;
      section	.appendChild ( myBlock ) ;
      
      const Publications = jsonObj[fieldtag] ;
      for ( let i = 0; i < Publications.length; i++ ) {
      	const myPubPart	= document .createElement('article');
        const TextTitle	= document .createElement('h4');
        const TextURL	= document .createElement('p');
        const PubDate	= document .createElement('p');
        TextTitle.textContent = Publications[i]['Title'];
        TextURL	 .textContent = Publications[i][ 'URL' ];
      	PubDate	 .textContent = Publications[i]['Date' ].split('T')[0];
      	myPubPart .appendChild ( TextTitle ) ;
      	myPubPart .appendChild ( TextURL ) ;
      	myPubPart .appendChild ( PubDate ) ;
      	section   .appendChild ( myPubPart ) ;
      }
    }

    function showOther( jsonObj, fieldtag ) {
      const section	= document .querySelector( '#other' );
      const myBlock	= document .createElement('block');
      Header	= document .createElement ( 'h2' ) ;
      Header	.textContent = fieldtag ;
      myBlock	.appendChild ( Header ) ;
      section	.appendChild ( myBlock ) ;
      const OtherStuffs = jsonObj[fieldtag] ;
      for ( let i = 0 ; i < OtherStuffs.length ; i++ ) {
      	const myStuff	= document .createElement('block');
        const TextTitle	= document .createElement('h4');
        singleKey	= Object.keys( OtherStuffs[i] )[0];
	TextTitle .textContent = singleKey + '\t\t'
		+ OtherStuffs[i][singleKey] ;
	myStuff .appendChild ( TextTitle ) ;
	section .appendChild ( myStuff ) ;
      }
    }

    function showFooter( jsonObj, fieldtag ) {
      const footer	= document .querySelector( '#footer' );
      const myBlock	= document .createElement('block');
      TodaysDate	= document .createElement('h6') ;
      TodaysDate	.textContent = Date() ;
      myBlock	.appendChild ( TodaysDate ) ;
      footer	.appendChild ( myBlock ) ;
    }

    function convertDateTime(DateTime) {
	var txt = Array();
	DateTime .forEach( DateTime2Date )
	function DateTime2Date(value, index, array) {
		txt.push( value.split('T')[0]) ;
	}
	return ( txt.join(' -- ') ) ;
    }

    function showLanguages ( jsonObj , fieldtag ) 
    {
      const section = document .querySelector( '#languages' );
      const myBlock = document .createElement('block');
      LangLife	= document .createElement('h2') ;
      LangLife	.textContent = fieldtag;
      myBlock	.appendChild ( LangLife ) ;
      section	.appendChild ( myBlock ) ;
      const Languages = jsonObj[fieldtag] ;
      for ( let i = 0; i < Languages.length; i++ ) {
      	const myLangPart = document .createElement('article');
        const TextLang = document .createElement('h4');
        const TextLevel = document .createElement('p');
        TextLang .textContent = Languages[i]['Type'] ;
        TextLevel.textContent = 'CEFR level: ' + Languages[i]['CEFR level'] ;
        	+ ' | ILR level: ' + Languages[i]['ILR level'];
	myLangPart .appendChild ( TextLang );
	myLangPart .appendChild ( TextLevel );	
	section .appendChild ( myLangPart );
      }
    }

    function showCertificates ( jsonObj , fieldtag ) 
    {
      const section = document .querySelector( '#certificates' );
      const myBlock = document .createElement('block');
      CertLife	= document .createElement('h2') ;
      CertLife	.textContent = fieldtag
      myBlock	.appendChild ( CertLife ) ;
      section	.appendChild ( myBlock ) ;

      const Certificates = jsonObj[fieldtag] ;
      for ( let i = 0; i < Certificates.length; i++ ) {
      	const myCertificatePart = document .createElement('article');
      	const TextCert		= document .createElement('h5');
        TextCert .textContent	= Certificates[i][ 'Date' ].split('T')[0];
        TextCert .textContent	= TextCert .textContent +'\t\t' + Certificates[i][ 'Type' ].split('T')[0];   
        myCertificatePart.appendChild ( TextCert );
        
      	const ValidFor		= document .createElement('p');
      	ValidFor .textContent	= Certificates[i][ 'Valid for' ] + '\t\t' + Certificates[i][ 'Issuer' ]
        myCertificatePart.appendChild ( ValidFor );
        section .appendChild ( myCertificatePart );
      }
    }
    
    function showEducation( jsonObj , fieldtag ) {
      const section = document .querySelector( '#education' );
      const myBlock = document .createElement('block');
      EduLife = document .createElement('h2') ;
      EduLife .textContent = fieldtag
      myBlock .appendChild ( EduLife ) ;
      section .appendChild ( myBlock ) ;
      
      const Educations = jsonObj[fieldtag] ;
      for ( let i = 0; i < Educations.length; i++ ) {
        const myEducationPart = document.createElement('article');
        const localHeadline = document.createElement('h3');

        localHeadline.textContent = Educations[i]['Degree'];
        
        const Dates		= document.createElement('h5');        
        Dates.textContent	= convertDateTime( Educations[i][ 'Dates' ] );
        
	const mySpecialisation	= document.createElement('p'); 
	mySpecialisation.textContent	= Educations[i]['Specialisation']
        const myField		= document.createElement('p');
        myField.textContent		= Educations[i]['Field of study']
        const mySchool		= document.createElement('p');
        mySchool.textContent		= Educations[i]['School']
        const myPlace		= document.createElement('p');
        myPlace.textContent		= Educations[i]['Place']
        
        const myDegree		= document.createElement('h4');
        if ( typeof Educations[i]['Degree publication'] !== 'undefined' ) {
		myDegree.textContent = Educations[i]['Degree publication'] ;
	}

        myEducationPart .appendChild ( localHeadline );
        myEducationPart .appendChild ( Dates );
        myEducationPart .appendChild ( myDegree );
        myEducationPart .appendChild ( mySpecialisation );
        myEducationPart .appendChild ( myField );
        myEducationPart .appendChild ( mySchool );
        myEducationPart .appendChild ( myPlace );        
        section.appendChild( myEducationPart );
      }
    }

    function showWork(jsonObj, fieldtag) {
      const section 	= document .querySelector( '#work' );
      WorkLife = document.createElement('h2') ;
      WorkLife .textContent = fieldtag
      section .appendChild( WorkLife );

      const Jobs = jsonObj[fieldtag];
      
      for ( let i = 0; i < Jobs.length; i++ ) {
        const myArticle		= document.createElement('article');
        const localHeadline	= document.createElement('h3');
        const CompanyHeader	= document.createElement('h4');
        const DescriptionHeader	= document.createElement('h4');
                
        const CompanyName	= document.createElement('p');
        const CompanyBranch	= document.createElement('p');
        const CompanyDivision	= document.createElement('p');
        const CompanyPlace	= document.createElement('p');
        const myDescription	= document.createElement('p');

        const ToolsTale		= document.createElement('p');
        const Dates		= document.createElement('h5');
        const SkillsTale	= document.createElement('p');
        const TailURL		= document.createElement('p');
        
        TailURL .textContent = Jobs[i]['URL'];
        
        localHeadline.textContent = Jobs[i].Position;

        CompanyHeader.textContent = Jobs[i]['Company']['Name'];
        DescriptionHeader.textContent = 'Description' ;
        
        CompanyName.textContent		= ''
        	+ Jobs[i]['Company']['Name'];
        CompanyBranch.textContent	= ''
        	+ Jobs[i]['Company']['Branch'];
        CompanyDivision.textContent	= ''
        	+ Jobs[i]['Company']['Division'];
        CompanyPlace.textContent	= ''
        	+ Jobs[i]['Place'];       	
        myDescription.textContent 	= ''
        	+ Jobs[i]['Description'];

        const ToolsList = document.createElement('ul');
        const superPowers = Jobs[i]['Tools'];
        for ( let j = 0; j < superPowers.length; j++ ) {
          const listItem = document.createElement('li');
          listItem.textContent = superPowers[j];
          ToolsList.appendChild(listItem);
        }
	//
	// GET AND CONVERT DATETIMES
	Dates.textContent 	= convertDateTime( Jobs[i][ 'Dates' ] );
	//
	// LINEAR SKILLS AND TOOLS STORY
        ToolsTale.textContent	= Jobs[i][ 'Tools' ].join(', ') ;
        SkillsTale.textContent	= Jobs[i][ 'Skills'].join(', ') ;        
	//
	// CREATE PROJECTS STRUCTURE
        const ProjectHeader	  = document.createElement('h4');
        ProjectHeader.textContent = 'Projects'
        //
        ttext = document.createElement( 'p' ) ;
	const Projects = Jobs[i][ 'Projects' ] ;
        const ProjectList = document.createElement('ul');
	for ( let i = 0; i < Projects.length; i++ ) {
		const Project	= Projects[i];
		const listItem	= document.createElement('li');
		listItem.textContent = 'Role:  '+Project['Role']
		listItem.textContent = listItem.textContent + '. Name: ' + Project['Name'] + '. Description: ' + Project['Description'] 
		var urls = Array() ;
		if ( typeof Project['URL'] !== 'undefined' ) {
			listItem.textContent = listItem.textContent + ' ' + Project['URL'] ;
		}
		if ( typeof( Project['URLS'] ) == typeof(Array()) ) {
			for ( iurl in Project['URLS'] )
			{
				if ( typeof( Project['URLS'][iurl] ) === typeof(String()) )	{
					listItem.textContent = listItem.textContent + ' ' + Project['URLS'][iurl] ;
				}
			}
		}
		if (urls.length>0) {
			ttext.textContent = ttext.textContent + urls.join(' | ')
		}
		ProjectList.appendChild(listItem);
	}
	//
	// PIECE IT TOGETHER
        myArticle .appendChild (	localHeadline	);
        myArticle .appendChild (	Dates		);
        myArticle .appendChild (	CompanyHeader	);
        myArticle .appendChild (	CompanyBranch	);
        myArticle .appendChild (	CompanyDivision	);
        myArticle .appendChild (	CompanyPlace	);        
        myArticle .appendChild ( DescriptionHeader	);        
        myArticle .appendChild (	myDescription	);
        myArticle .appendChild (	TailURL		);
        myArticle .appendChild (	ProjectHeader	);
        myArticle .appendChild (	ProjectList	);        
        myArticle .appendChild (	ttext		);
        myArticle .appendChild (	ToolsTale	);
        myArticle .appendChild (	SkillsTale	);
        
        section.appendChild(myArticle);
      }
    }
